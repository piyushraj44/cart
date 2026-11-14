"""v2 experiments: Mathur-augmented training data and a fine-tuned transformer, compared with the frozen served model.

Run from the project root (after `.venv/bin/python -m ml.mathur`):

    .venv/bin/python -m ml.train_v2              # trains what is missing, evaluates everything, writes metrics
    .venv/bin/python -m ml.train_v2 --retrain    # re-fine-tune the transformers even if saved
    .venv/bin/python -m ml.train_v2 --redo-cv    # recompute the repeated grouped CV

Never touches ml/artifacts/* or data/processed/text_dataset.csv. Outputs:
    ml/artifacts_v2/ensemble_v2/clearcart_text_model.joblib
    ml/artifacts_v2/transformer_v2/, ml/artifacts_v2/transformer_v1data/   (HF model + clearcart_config.json + neighbors.npz)
    ml/artifacts_v2/metrics.json, ml/artifacts_v2/selected.json, ml/artifacts_v2/cv_cache.json
    report/figures/v2_*.png

Protocol (identical for every model)
    train on TRAIN (+ eligible Mathur rows for *_v2) -> choose hyper-parameters / early-stopping epoch on VAL ->
    temperature T fit on VAL NLL -> tau chosen on VAL with ml/train.py's rule -> TEST evaluated once.
    Final numbers for every model are computed by reloading the SAVED artefact through the serving class
    (ml.model.load_classifier), so metrics.json describes exactly what would be served.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix, f1_score, log_loss,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from ml import train as T1
from ml.common import ARTIFACTS, DATA_PROCESSED, EMBEDDER_NAME, FIGURES, LABELS, ROOT, SEED
from ml.mathur import ec_with_domains, git_commit, load_mathur
from ml.model import (ClassifierBundle, TextClassifier, load_classifier, softmax, surface_features,
                      temperature_scale)
from ml.preprocess import UnionFind

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
V2 = ROOT / "ml" / "artifacts_v2"
L2I, BENIGN = T1.L2I, T1.BENIGN
DARK_COLS = [i for i in range(len(LABELS)) if i != BENIGN]
TRANSFORMER = "distilroberta-base"
HP = {"lr": 3e-5, "batch_size": 16, "weight_decay": 0.01, "warmup_ratio": 0.1, "max_epochs": 5, "patience": 2,
      "max_length": 128, "grad_clip": 1.0, "loss": "class-weighted cross-entropy (sklearn 'balanced' weights)"}
PROBES = ["@ Time left 11h 01m", "FREE SHIPPING FOR A LIMITED PERIOD +", "Sale ends Sunday 23:59 GMT"]
DEADLINE_RE = re.compile(r"\b(ends?|ending|left|hours?|hrs?|minutes?|mins?|today|tonight|tomorrow|limited|time|until|"
                         r"expires?|deadline|sale|offer|days?|week|sunday|monday|tuesday|wednesday|thursday|friday|saturday)\b"
                         r"|\d{1,2}\s*:\s*\d{2}", re.I)


def r4(x) -> float:
    return round(float(x), 4)


# ---------------------------------------------------------------- data
def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = ec_with_domains()
    df["y"] = df["label"].map(L2I)
    df["source"] = "ec-darkpattern"
    mr = pd.read_csv(DATA_PROCESSED / "mathur_rows.csv", dtype=str, keep_default_na=False)
    el = mr[mr["eligible"] == "True"]
    aug = pd.DataFrame({"text": el["text"].values, "label": el["label"].values, "page_id": ("m" + el["mathur_row"]).values,
                        "template_key": el["template_key"].values, "domain": el["domain"].values, "split": "train",
                        "source": "mathur", "mathur_row": el["mathur_row"].astype(int).values})
    aug["y"] = aug["label"].map(L2I)
    return df, aug


# ---------------------------------------------------------------- decision rules (same as ml/train.py)
def select_tau(yva: np.ndarray, pdark_val: np.ndarray) -> tuple[float, str]:
    curve = []
    for t in np.round(np.arange(0.10, 0.96, 0.01), 2):
        b = T1.binary_block(yva, pdark_val, t)
        curve.append((float(t), b["precision"], b["recall"], b["f1"]))
    ok = [c for c in curve if c[1] >= 0.90]
    if not ok:
        best = max(c[3] for c in curve)
        return max(c[0] for c in curve if c[3] >= best - 0.005), "fallback: no tau reached precision>=0.90 on VAL"
    best = max(c[3] for c in ok)
    return max(c[0] for c in ok if c[3] >= best - 0.005), "max VAL F1 s.t. precision>=0.90, near-ties -> higher tau"


def app_pred(p: np.ndarray, tau: float) -> np.ndarray:
    pdark = 1 - p[:, BENIGN]
    return np.where(pdark >= tau, np.array(DARK_COLS)[p[:, DARK_COLS].argmax(1)], BENIGN)


def eval_split(y: np.ndarray, raw: np.ndarray, T: float, tau: float, abstained: np.ndarray | None = None) -> dict:
    cal = temperature_scale(raw, T)
    pred = app_pred(cal, tau)
    rep = classification_report(y, pred, labels=list(range(len(LABELS))), target_names=LABELS, output_dict=True, zero_division=0)
    b = T1.binary_block(y, 1 - cal[:, BENIGN], tau)
    out = {
        "n": int(len(y)),
        "accuracy": r4((pred == y).mean()),
        "macro_f1": r4(f1_score(y, pred, average="macro", labels=list(range(len(LABELS))))),
        "per_class_f1": {l: r4(rep[l]["f1-score"]) for l in LABELS},
        "per_class": {l: {k: r4(rep[l][k]) for k in ("precision", "recall", "f1-score", "support")} for l in LABELS},
        "dark_vs_benign": b,
        "benign_false_positive_rate": b["false_positive_rate"],
        "roc_auc_ovr_macro": r4(roc_auc_score(y, cal, multi_class="ovr", average="macro", labels=list(range(len(LABELS)))))
        if len(set(y)) == len(LABELS) else None,
        "ece_before_calibration": r4(T1.ece(y, raw)),
        "ece_after_calibration": r4(T1.ece(y, cal)),
        "log_loss_before": r4(log_loss(y, raw, labels=list(range(len(LABELS))))),
        "log_loss_after": r4(log_loss(y, cal, labels=list(range(len(LABELS))))),
        "brier_after": r4(T1.multiclass_brier(y, cal)),
        "argmax_macro_f1": r4(f1_score(y, cal.argmax(1), average="macro")),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(len(LABELS)))).tolist(),
    }
    if abstained is not None:
        pa = np.where(abstained, BENIGN, pred)
        out["macro_f1_with_abstention_gate"] = r4(f1_score(y, pa, average="macro", labels=list(range(len(LABELS)))))
    return out


def evaluate(clf, df: pd.DataFrame, train_domains: set) -> dict:
    """Score a loaded serving classifier on frozen VAL and TEST (T and tau taken from the classifier)."""
    from ml.model import has_lexical_evidence
    res = {"temperature": r4(clf.b.temperature), "threshold": float(clf.b.threshold)}
    raws = {}
    for s in ("val", "test"):
        d = df[df["split"] == s]
        raw, _ = clf.raw_proba(d["text"].tolist())
        raws[s] = raw
        ab = ~d["text"].map(has_lexical_evidence).values
        res[s] = eval_split(d["y"].values, raw, clf.b.temperature, clf.b.threshold, ab)
    d = df[df["split"] == "test"]
    unseen = ~d["domain"].isin(train_domains).values
    res["test_domain_unseen_subset"] = {"note": "TEST rows whose website domain has no row in this model's training data",
                                        **eval_split(d["y"].values[unseen], raws["test"][unseen], clf.b.temperature, clf.b.threshold)}
    for k in ("confusion_matrix", "per_class"):
        res["test_domain_unseen_subset"].pop(k)
    return res, raws


# ---------------------------------------------------------------- ensemble recipe (reuses ml/train.py)
def train_ensemble(tr: pd.DataFrame, va: pd.DataFrame, Etr, Eva) -> tuple[ClassifierBundle, dict]:
    ytr, yva = tr["y"].values, va["y"].values
    Xtr, Xva = tr["text"].tolist(), va["text"].tolist()
    vec = T1.make_tfidf().fit(Xtr)
    Ttr, Tva = vec.transform(Xtr), vec.transform(Xva)
    grid, best = {}, None
    for C in [0.5, 1, 2, 4, 8, 16]:
        m = T1.make_lr(C).fit(Ttr, ytr)
        s = f1_score(yva, m.predict(Tva), average="macro"); grid[f"tfidf_lr_C={C}"] = r4(s)
        if best is None or s > best[0]:
            best = (s, C, m)
    C_lr, lr = best[1], best[2]
    Str, Sva = surface_features(Xtr), surface_features(Xva)
    xgb = T1.make_xgb().fit(np.hstack([Etr, Str]), ytr, sample_weight=compute_sample_weight("balanced", ytr),
                            eval_set=[(np.hstack([Eva, Sva]), yva)], verbose=False)
    n_trees = int(xgb.best_iteration) + 1
    p_lr, p_x = lr.predict_proba(Tva), xgb.predict_proba(np.hstack([Eva, Sva]))
    best = None
    for w in np.round(np.arange(0, 1.01, 0.1), 2):
        s = f1_score(yva, (w * p_lr + (1 - w) * p_x).argmax(1), average="macro"); grid[f"ensemble_w_lr={w}"] = r4(s)
        if best is None or s > best[0] + 1e-9:
            best = (s, float(w))
    w = best[1]
    members = {"tfidf_lr": f1_score(yva, p_lr.argmax(1), average="macro"), "minilm_xgb": f1_score(yva, p_x.argmax(1), average="macro"),
               "ensemble_lr_xgb": best[0]}
    selected = max(members, key=members.get)
    if selected == "tfidf_lr":
        w = 1.0
    elif selected == "minilm_xgb":
        w = 0.0
    p_val_raw = w * p_lr + (1 - w) * p_x
    T = T1.fit_temperature(yva, p_val_raw)
    tau, rule = select_tau(yva, 1 - temperature_scale(p_val_raw, T)[:, BENIGN])
    nb = tr.drop_duplicates("text")
    nb_emb = Etr[~tr["text"].duplicated().values]
    bundle = ClassifierBundle(
        labels=LABELS, tfidf=vec, lr=lr, xgb=xgb, weight_lr=w, temperature=T, threshold=tau,
        neighbor_texts=nb["text"].tolist(), neighbor_labels=nb["label"].tolist(), neighbor_emb=nb_emb.astype(np.float32),
        meta={"embedder": EMBEDDER_NAME, "selected_model": "ensemble_v2",
              "trained_on": "ec-darkpattern TRAIN split + eligible Mathur et al. 2019 rows (see data/processed/mathur_inspection.md)",
              "n_train": int(len(tr)), "n_train_mathur": int((tr["source"] == "mathur").sum()), "member_selected_on_val": selected,
              "lr_C": C_lr, "xgb_trees": n_trees, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "version": "v2"})
    info = {"val_member_macro_f1_argmax": {k: r4(v) for k, v in members.items()}, "member_selected": selected, "weight_lr": w,
            "lr_C": C_lr, "xgb_trees": n_trees, "temperature": r4(T), "tau": tau, "tau_rule": rule, "grid_val_macro_f1": grid}
    return bundle, info


# ---------------------------------------------------------------- transformer fine-tuning
def device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _logits(model, tok, texts, dev, bs=64):
    import torch
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], truncation=True, max_length=HP["max_length"], padding=True, return_tensors="pt").to(dev)
            out.append(model(**enc).logits.float().cpu().numpy())
    return np.vstack(out)


def finetune(tr_texts, tr_y, va_texts=None, va_y=None, seed: int = SEED, fixed_epochs: int | None = None, log=print):
    """Fine-tune distilroberta-base. Early stopping on VAL macro-F1 (argmax) unless fixed_epochs is given."""
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup,
                              set_seed)
    set_seed(seed)
    dev = device()
    tok = AutoTokenizer.from_pretrained(TRANSFORMER)
    model = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=L2I).to(dev)
    cw = torch.tensor(compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=tr_y), dtype=torch.float32, device=dev)
    no_decay = ("bias", "LayerNorm.weight")
    params = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": HP["weight_decay"]},
              {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(params, lr=HP["lr"])
    n_ep = fixed_epochs or HP["max_epochs"]
    steps = math.ceil(len(tr_texts) / HP["batch_size"]) * n_ep
    sch = get_linear_schedule_with_warmup(opt, int(HP["warmup_ratio"] * steps), steps)
    rng = np.random.RandomState(seed)
    tr_texts, tr_y = list(tr_texts), np.asarray(tr_y)
    hist, best, best_state, bad = [], -1.0, None, 0
    for ep in range(1, n_ep + 1):
        model.train(); t0 = time.time(); tot = 0.0
        order = rng.permutation(len(tr_texts))
        for i in range(0, len(order), HP["batch_size"]):
            idx = order[i:i + HP["batch_size"]]
            enc = tok([tr_texts[j] for j in idx], truncation=True, max_length=HP["max_length"], padding=True, return_tensors="pt").to(dev)
            yb = torch.tensor(tr_y[idx], dtype=torch.long, device=dev)
            loss = torch.nn.functional.cross_entropy(model(**enc).logits, yb, weight=cw)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), HP["grad_clip"])
            opt.step(); sch.step(); tot += float(loss) * len(idx)
        rec = {"epoch": ep, "train_loss": r4(tot / len(tr_texts)), "seconds": round(time.time() - t0, 1)}
        if va_texts is not None and fixed_epochs is None:
            s = f1_score(va_y, _logits(model, tok, list(va_texts), dev).argmax(1), average="macro")
            rec["val_macro_f1_argmax"] = r4(s)
            if s > best + 1e-9:
                best, bad = s, 0
                best_state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
                rec["best"] = True
            else:
                bad += 1
        hist.append(rec); log(f"    {rec}")
        if va_texts is not None and fixed_epochs is None and bad >= HP["patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    best_epoch = max((h for h in hist if h.get("best")), key=lambda h: h["epoch"])["epoch"] if best_state is not None else n_ep
    return model, tok, hist, best_epoch


def build_transformer(name: str, tr: pd.DataFrame, va: pd.DataFrame, retrain: bool) -> dict:
    """Fine-tune (or reuse), save, then calibrate T and tau on VAL through the CPU serving class."""
    import torch
    from ml.model import TransformerClassifier
    out_dir = V2 / name
    cfg_path = out_dir / "clearcart_config.json"
    if retrain or not (out_dir / "model.safetensors").exists() or not cfg_path.exists():
        print(f"[{name}] fine-tuning on {len(tr)} rows ({device()})")
        t0 = time.time()
        model, tok, hist, best_epoch = finetune(tr["text"].tolist(), tr["y"].values, va["text"].tolist(), va["y"].values)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.to("cpu").save_pretrained(str(out_dir)); tok.save_pretrained(str(out_dir))
        meta = {"selected_model": name, "version": "v2", "base_model": TRANSFORMER, "hyperparameters": HP, "seed": SEED,
                "train_device": device(), "history": hist, "best_epoch": best_epoch, "n_train": int(len(tr)),
                "n_train_mathur": int((tr["source"] == "mathur").sum()),
                "trained_on": ("ec-darkpattern TRAIN split + eligible Mathur et al. 2019 rows" if name.endswith("_v2")
                               else "ec-darkpattern TRAIN split only"),
                "train_seconds": round(time.time() - t0, 1), "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "embedder": f"{TRANSFORMER} (fine-tuned)", "inference_device": "cpu"}
        cfg_path.write_text(json.dumps({"labels": LABELS, "temperature": 1.0, "threshold": 0.5, "max_length": HP["max_length"], "meta": meta}, indent=2))
        del model
        if device() == "mps":
            torch.mps.empty_cache()
        np.savez(out_dir / "neighbors.npz", texts=np.array([], dtype=str), labels=np.array([], dtype=str), emb=np.zeros((0, 768), np.float32))
    cfg = json.loads(cfg_path.read_text())
    clf = TransformerClassifier.load(out_dir)
    # neighbour bank = de-duplicated training texts, embedded with the fine-tuned encoder on CPU
    nb = tr.drop_duplicates("text")
    _, nb_emb = clf.forward(nb["text"].tolist())
    np.savez(out_dir / "neighbors.npz", texts=np.array(nb["text"].tolist(), dtype=str), labels=np.array(nb["label"].tolist(), dtype=str),
             emb=nb_emb.astype(np.float32))
    raw_val, _ = clf.raw_proba(va["text"].tolist())
    T = T1.fit_temperature(va["y"].values, raw_val)
    tau, rule = select_tau(va["y"].values, 1 - temperature_scale(raw_val, T)[:, BENIGN])
    cfg["temperature"], cfg["threshold"] = T, tau
    cfg["meta"]["tau_rule"] = rule
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return {"best_epoch": cfg["meta"]["best_epoch"], "history": cfg["meta"]["history"], "temperature": r4(T), "tau": tau, "tau_rule": rule,
            "n_train": cfg["meta"]["n_train"], "n_train_mathur": cfg["meta"]["n_train_mathur"], "train_seconds": cfg["meta"]["train_seconds"],
            "train_device": cfg["meta"]["train_device"]}


# ---------------------------------------------------------------- repeated grouped CV on train+val
def repeated_cv(recipe: str, data: str, df: pd.DataFrame, aug: pd.DataFrame, emb_lookup: dict, params: dict,
                seeds=(0, 1, 2), k: int = 5) -> dict:
    pool = df[df["split"] != "test"].reset_index(drop=True)
    scores = []
    for seed in seeds:
        sg = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        for fi, (tri, tei) in enumerate(sg.split(pool, pool["y"], groups=pool["group_id"])):
            tr, te = pool.iloc[tri], pool.iloc[tei]
            n_aug = 0
            if data == "v2":
                a = aug[~aug["template_key"].isin(set(te["template_key"])) & ~aug["domain"].isin(set(te["domain"]))
                        & ~aug["text"].isin(set(te["text"]))]
                n_aug = len(a)
                tr = pd.concat([tr, a], ignore_index=True)
            if recipe == "ensemble":
                vec = T1.make_tfidf().fit(tr["text"])
                lr = T1.make_lr(params["lr_C"]).fit(vec.transform(tr["text"]), tr["y"])
                Etr = np.vstack([emb_lookup[t] for t in tr["text"]]); Ete = np.vstack([emb_lookup[t] for t in te["text"]])
                xgb = T1.make_xgb(params["xgb_trees"], early=False).fit(
                    np.hstack([Etr, surface_features(tr["text"].tolist())]), tr["y"], sample_weight=compute_sample_weight("balanced", tr["y"]))
                w = params["weight_lr"]
                p = w * lr.predict_proba(vec.transform(te["text"])) + (1 - w) * xgb.predict_proba(np.hstack([Ete, surface_features(te["text"].tolist())]))
            else:
                model, tok, _, _ = finetune(tr["text"].tolist(), tr["y"].values, seed=SEED + seed, fixed_epochs=params["epochs"], log=lambda *_: None)
                p = softmax(_logits(model, tok, te["text"].tolist(), device()))
                del model
            s = f1_score(te["y"], p.argmax(1), average="macro")
            scores.append(r4(s))
            print(f"    CV {recipe}/{data} seed={seed} fold={fi} n_aug={n_aug} macro-F1={s:.4f}", flush=True)
    return {"mean_macro_f1": r4(np.mean(scores)), "std": r4(np.std(scores)), "n_folds": len(scores), "fold_scores": scores,
            "protocol": f"StratifiedGroupKFold({k}) x seeds {list(seeds)} on ec-darkpattern TRAIN+VAL (frozen group_id); argmax macro-F1 on "
                        f"held-out ec rows; {'eligible Mathur rows added to the training folds only, minus any sharing text/template/domain with the held-out fold' if data == 'v2' else 'no Mathur rows'}",
            "params": params}


# ---------------------------------------------------------------- Mathur `Deceptive?` model
def deceptive_model() -> dict:
    m = load_mathur()
    d = m[(m["text"] != "") & m["Deceptive?"].isin(["Yes", "No"])].reset_index(drop=True)
    y = (d["Deceptive?"] == "Yes").astype(int).values
    uf = UnionFind()
    for dom, tk in zip(d["domain"], d["template_key"]):
        uf.union(("d", dom), ("t", tk))
    groups = np.array([str(uf.find(("d", x))) for x in d["domain"]])
    per_seed, oof_all = [], []
    for seed in (0, 1, 2):
        oof, prior = np.zeros(len(d)), np.zeros(len(d))
        for tri, tei in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(d, y, groups):
            vec = T1.make_tfidf().fit(d["text"].iloc[tri])
            lr = T1.make_lr(1.0).fit(vec.transform(d["text"].iloc[tri]), y[tri])
            oof[tei] = lr.predict_proba(vec.transform(d["text"].iloc[tei]))[:, 1]
            rate = pd.Series(y[tri]).groupby(d["Pattern Type"].iloc[tri].values).mean()
            prior[tei] = d["Pattern Type"].iloc[tei].map(rate).fillna(y[tri].mean()).values
        pred = (oof >= 0.5).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        cd = (d["Pattern Type"] == "Countdown Timer").values
        per_seed.append({"auroc": roc_auc_score(y, oof), "average_precision": average_precision_score(y, oof),
                         "precision_at_0.5": p, "recall_at_0.5": r, "f1_at_0.5": f,
                         "pattern_type_prior_auroc": roc_auc_score(y, prior),
                         "auroc_within_countdown_timer": roc_auc_score(y[cd], oof[cd]),
                         "auroc_excluding_countdown_timer": roc_auc_score(y[~cd], oof[~cd])})
        oof_all.append(oof)
    agg = {k: {"mean": r4(np.mean([s[k] for s in per_seed])), "std": r4(np.std([s[k] for s in per_seed]))} for k in per_seed[0]}
    cd = (d["Pattern Type"] == "Countdown Timer").values
    return {"task": "Among Mathur dark patterns with text: predict Deceptive?=Yes (verified fake) vs No (real pressure, not proven fake)",
            "n": int(len(d)), "n_yes": int(y.sum()), "prevalence_yes": r4(y.mean()), "n_groups": int(len(set(groups))),
            "n_countdown_timer": int(cd.sum()), "n_yes_countdown_timer": int(y[cd].sum()),
            "excluded": "Deceptive?=Depends (26 rows, all in the Sneaking category) and rows without Pattern String",
            "model": "TF-IDF (word 1-2g + char_wb 2-5g) + Logistic Regression (C=1, balanced)",
            "protocol": "StratifiedGroupKFold(5) x 3 seeds; groups = union of website domain and template key",
            "baseline": "pattern_type_prior: in-fold Yes-rate of the row's Mathur Pattern Type (uses a label the product does not observe)",
            "grouped_cv": agg}


# ---------------------------------------------------------------- figures
def fig_comparison(res: dict, path) -> None:
    names = list(res)
    x = np.arange(len(names))
    v = [res[n]["val"]["macro_f1"] for n in names]; t = [res[n]["test"]["macro_f1"] for n in names]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(x - 0.18, v, 0.36, label="VAL macro-F1 (selection)", color="#2563eb")
    ax.bar(x + 0.18, t, 0.36, label="TEST macro-F1 (held out)", color="#94a3b8")
    for i in range(len(names)):
        ax.text(i - 0.18, v[i] + 0.004, f"{v[i]:.3f}", ha="center", fontsize=7)
        ax.text(i + 0.18, t[i] + 0.004, f"{t[i]:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x, [n.replace("_", "\n", 1) for n in names], fontsize=8)
    ax.set_ylim(max(0, min(v + t) - 0.08), 1.0); ax.set_ylabel("Macro-F1 (application decision)")
    ax.legend(fontsize=7, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_cv(cv: dict, path) -> None:
    names = list(cv)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.boxplot([cv[n]["fold_scores"] for n in names], showmeans=True)
    ax.set_xticks(range(1, len(names) + 1), [n.replace("/", "\n") for n in names], fontsize=8)
    ax.set_ylabel("Fold macro-F1 (argmax)"); ax.set_title("Repeated grouped CV on TRAIN+VAL", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--redo-cv", action="store_true")
    ap.add_argument("--skip-cv", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    V2.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    df, aug = load_all()
    tr1 = df[df["split"] == "train"]
    va, te = df[df["split"] == "val"], df[df["split"] == "test"]
    tr2 = pd.concat([tr1, aug], ignore_index=True)
    dom1, dom2 = set(tr1["domain"]), set(tr2["domain"])
    metrics: dict = {"seed": SEED, "labels": LABELS, "created_by": "ml/train_v2.py",
                     "data": {"train_v1_rows": int(len(tr1)), "train_v2_rows": int(len(tr2)), "mathur_rows_added": int(len(aug)),
                              "mathur_rows_added_by_label": {k: int(v) for k, v in aug["label"].value_counts().items()},
                              "mathur_unique_texts_added": int(aug["text"].nunique()),
                              "mathur_texts_not_already_in_train": int((~aug["text"].isin(set(tr1["text"]))).sum()),
                              "val_rows": int(len(va)), "test_rows": int(len(te)), "mathur_commit": git_commit(),
                              "frozen_inputs": "data/processed/text_dataset.csv split column (unchanged)"}}
    print("data:", metrics["data"])

    # ---- embeddings (frozen MiniLM from ml/artifacts/minilm, read only)
    embedder = T1.get_embedder()
    all_texts = list(dict.fromkeys(df["text"].tolist() + aug["text"].tolist()))
    E = T1.embed(embedder, all_texts)
    emb_lookup = dict(zip(all_texts, E))
    emb_of = lambda d: np.vstack([emb_lookup[t] for t in d["text"]])  # noqa: E731

    # ---- ensemble_v2
    bundle, ens_info = train_ensemble(tr2, va, emb_of(tr2), emb_of(va))
    (V2 / "ensemble_v2").mkdir(exist_ok=True)
    joblib.dump(bundle, V2 / "ensemble_v2" / "clearcart_text_model.joblib", compress=3)
    print("ensemble_v2:", {k: v for k, v in ens_info.items() if k != "grid_val_macro_f1"})

    # ---- transformers
    tf_info = {"transformer_v1data": build_transformer("transformer_v1data", tr1, va, args.retrain),
               "transformer_v2": build_transformer("transformer_v2", tr2, va, args.retrain)}
    for k, v in tf_info.items():
        print(k, {kk: vv for kk, vv in v.items() if kk != "history"})

    # ---- evaluate every model through its serving loader
    models = {"served_v1": ("v1", dom1), "ensemble_v2": ("ensemble_v2", dom2),
              "transformer_v1data": ("transformer_v1data", dom1), "transformer_v2": ("transformer_v2", dom2)}
    res, clfs = {}, {}
    for name, (key, doms) in models.items():
        clf = load_classifier(key)
        clfs[name] = clf
        r, _ = evaluate(clf, df, doms)
        r["meta"] = {k: v for k, v in clf.b.meta.items() if k != "history"}
        res[name] = r
        print(f"{name}: VAL macro-F1 {r['val']['macro_f1']}  TEST macro-F1 {r['test']['macro_f1']}  "
              f"TEST benign FPR {r['test']['benign_false_positive_rate']}  ECE {r['test']['ece_before_calibration']}->{r['test']['ece_after_calibration']}")
    ref = json.loads((ARTIFACTS / "metrics.json").read_text())["final_test"]["application_metrics"]["macro_f1"]
    res["served_v1"]["reference_metrics_json_test_macro_f1"] = ref
    res["served_v1"]["recomputed_matches_reference"] = abs(res["served_v1"]["test"]["macro_f1"] - ref) <= 0.0005
    metrics["training"] = {"ensemble_v2": ens_info, **tf_info}
    metrics["comparison"] = res

    # ---- selection on VAL only
    new = ["ensemble_v2", "transformer_v1data", "transformer_v2"]
    val_scores = {n: res[n]["val"]["macro_f1"] for n in res}
    winner = max(new, key=lambda n: (val_scores[n], -new.index(n)))
    metrics["selection"] = {"rule": "highest VAL macro-F1 (application decision: calibrated P(dark)>=tau, argmax dark category); TEST not consulted",
                            "val_macro_f1": val_scores, "selected_v2_model": winner,
                            "winner_beats_served_v1_on_val": bool(val_scores[winner] > val_scores["served_v1"]),
                            "ranking_by_val": sorted(val_scores, key=val_scores.get, reverse=True)}
    (V2 / "selected.json").write_text(json.dumps({"selected_model": winner, "val_macro_f1": val_scores[winner],
                                                  "rule": metrics["selection"]["rule"], "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    print("selection:", metrics["selection"])

    # ---- repeated grouped CV for the two best by VAL
    cache_p = V2 / "cv_cache.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() and not args.redo_cv else {}
    # the two best by VAL (the rule asked for), plus both transformers so the model effect and the data effect
    # can also be read from CV (the ensemble recipes tie on VAL, so CV on them alone says nothing about the transformer)
    top2 = metrics["selection"]["ranking_by_val"][:2]
    cv_models = top2 + [n for n in ("transformer_v1data", "transformer_v2") if n not in top2]
    metrics["cv_models"] = {"top2_by_val": top2, "evaluated": cv_models}
    recipe_of = {"served_v1": ("ensemble", "v1"), "ensemble_v2": ("ensemble", "v2"),
                 "transformer_v1data": ("transformer", "v1"), "transformer_v2": ("transformer", "v2")}
    cv = {}
    if not args.skip_cv:
        for n in cv_models:
            recipe, data = recipe_of[n]
            if recipe == "ensemble":
                b = clfs[n].b
                params = {"lr_C": b.meta["lr_C"], "weight_lr": b.weight_lr, "xgb_trees": b.meta["xgb_trees"]}
            else:
                params = {"epochs": tf_info[n]["best_epoch"], **{k: HP[k] for k in ("lr", "batch_size", "max_length")}}
            key = f"{n}:{json.dumps(params, sort_keys=True)}"
            if key not in cache:
                print(f"[CV] {n} {params}")
                cache[key] = repeated_cv(recipe, data, df, aug, emb_lookup, params)
                cache_p.write_text(json.dumps(cache, indent=1))
            cv[n] = cache[key]
        fig_cv({f"{n}": cv[n] for n in cv}, FIGURES / "v2_cv.png")
    metrics["repeated_grouped_cv"] = cv
    metrics["v1_reference_cv_from_ml_artifacts_metrics_json"] = json.loads((ARTIFACTS / "metrics.json").read_text()).get("E5_repeated_group_cv")

    # ---- Deceptive? research model
    metrics["deceptive_model"] = deceptive_model()
    print("deceptive:", json.dumps(metrics["deceptive_model"]["grouped_cv"], indent=0))

    # ---- error analysis: benign false positives on TEST, deadline-like benign rows, user-reported probes
    te_r = te.reset_index(drop=True)
    benign = te_r[te_r["label"] == "benign"]
    deadline = benign[benign["text"].map(lambda s: bool(DEADLINE_RE.search(s)))]
    ea = {"deadline_regex": DEADLINE_RE.pattern, "test_benign_rows": int(len(benign)), "test_benign_deadline_like_rows": int(len(deadline)),
          "per_model": {}, "probes_note": "User-reported strings. Qualitative probes only: NOT test data, no ground-truth label, not in any metric.",
          "probes": {}}
    fp_sets = {}
    for name, clf in clfs.items():
        preds = clf.predict(benign["text"].tolist())
        fl = [p["flagged"] for p in preds]
        fps = benign[np.array(fl)]
        fp_sets[name] = set(fps["text"])
        dl = deadline["text"].isin(fps["text"])
        ea["per_model"][name] = {"benign_fp": int(len(fps)), "benign_fp_deadline_like": int(dl.sum()),
                                 "fp_texts": sorted(fps["text"].tolist())[:40],
                                 "deadline_like_fp_texts": sorted(deadline.loc[dl, "text"].tolist())}
        ea["probes"][name] = [{"text": p["text"], "p_dark": r4(p["p_dark"]), "flagged": bool(p["flagged"]), "category_if_flagged": p["category"],
                               "abstained": p["abstained"], "key_terms": clf.key_terms(p["text"], p["category"])} for p in clf.predict(PROBES)]
    for n in new:
        ea["per_model"][n]["fp_fixed_vs_served_v1"] = sorted(fp_sets["served_v1"] - fp_sets[n])
        ea["per_model"][n]["fp_introduced_vs_served_v1"] = sorted(fp_sets[n] - fp_sets["served_v1"])
    ea["deadline_like_benign_texts"] = sorted(deadline["text"].tolist())
    metrics["error_analysis"] = ea

    # ---- figures
    fig_comparison(res, FIGURES / "v2_model_comparison.png")
    for n in new:
        T1.fig_confusion(np.array(res[n]["test"]["confusion_matrix"]), FIGURES / f"v2_confusion_{n}.png",
                         f"{n}: held-out TEST confusion (n={len(te)})")
    raw_te, _ = clfs[winner].raw_proba(te["text"].tolist())
    T1.fig_reliability(te["y"].values, raw_te, temperature_scale(raw_te, clfs[winner].b.temperature), FIGURES / f"v2_reliability_{winner}.png")

    metrics["runtime_seconds"] = round(time.time() - t_start, 1)
    (V2 / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=float))
    rows = []
    for n, r in res.items():
        rows.append({"model": n, "VAL_mF1": r["val"]["macro_f1"], "TEST_acc": r["test"]["accuracy"], "TEST_mF1": r["test"]["macro_f1"],
                     **{f"F1_{l}": r["test"]["per_class_f1"][l] for l in LABELS},
                     "dark_P": r["test"]["dark_vs_benign"]["precision"], "dark_R": r["test"]["dark_vs_benign"]["recall"],
                     "dark_F1": r["test"]["dark_vs_benign"]["f1"], "benign_FPR": r["test"]["benign_false_positive_rate"],
                     "ECE_raw": r["test"]["ece_before_calibration"], "ECE_cal": r["test"]["ece_after_calibration"],
                     "unseen_dom_mF1": r["test_domain_unseen_subset"]["macro_f1"]})
    print(pd.DataFrame(rows).to_string(index=False))
    print("winner:", winner, "runtime", metrics["runtime_seconds"])


if __name__ == "__main__":
    main()
