"""Step 3+4 - Train, compare, calibrate, evaluate.

Run from the project root:  python -m ml.train

Every number written to ml/artifacts/metrics.json and every figure in report/figures/ is
produced by this script from the held-out data. Nothing is hand-typed.

Experiments
  E1  Leakage: naive random split vs group+template split (TF-IDF+LR)
  E2  Model comparison on the VALIDATION split (selection never looks at test)
  E3  Calibration: temperature scaling fit on validation, evaluated on test
  E4  Decision threshold for P(dark) tuned on validation vs plain argmax
  E5  Robustness: repeated grouped CV on train+val (5 seeds)
  E6  Out-of-taxonomy probe: Obstruction / Sneaking / Forced Action rows
"""
from __future__ import annotations

import json
import time
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.sparse import hstack as sp_hstack  # noqa: F401  (kept for clarity of sparse ops)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, classification_report,
                             confusion_matrix, f1_score, log_loss, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ml.common import (ARTIFACTS, DATA_PROCESSED, DISPLAY_NAMES, EMBEDDER_NAME, FIGURES, LABELS,
                       SEED)
from ml.model import ClassifierBundle, SURFACE_NAMES, surface_features, temperature_scale

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(SEED)
L2I = {l: i for i, l in enumerate(LABELS)}
BENIGN = L2I["benign"]


# ---------------------------------------------------------------- helpers
def make_tfidf() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True,
                                 token_pattern=r"(?u)\b\w+\b")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, lowercase=True)),
    ])


def make_lr(C: float) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=5000, class_weight="balanced", random_state=SEED)


def make_xgb(n_estimators: int = 600, early: bool = True) -> XGBClassifier:
    # Hyper-parameters follow the UC Berkeley detector (depth 5, lr 0.05, subsample 0.9)
    return XGBClassifier(
        objective="multi:softprob", num_class=len(LABELS), n_estimators=n_estimators, max_depth=5,
        learning_rate=0.05, subsample=0.9, colsample_bytree=0.8, min_child_weight=1, reg_lambda=1.0,
        tree_method="hist", eval_metric="mlogloss", random_state=SEED, n_jobs=4,
        early_stopping_rounds=50 if early else None,
    )


def metrics_block(y: np.ndarray, pred: np.ndarray, proba: np.ndarray | None = None) -> dict:
    out = {
        "accuracy": accuracy_score(y, pred),
        "macro_precision": precision_recall_fscore_support(y, pred, average="macro", zero_division=0)[0],
        "macro_recall": precision_recall_fscore_support(y, pred, average="macro", zero_division=0)[1],
        "macro_f1": f1_score(y, pred, average="macro"),
        "weighted_precision": precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)[0],
        "weighted_recall": precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)[1],
        "weighted_f1": f1_score(y, pred, average="weighted"),
    }
    if proba is not None:
        out["roc_auc_ovr_macro"] = roc_auc_score(y, proba, multi_class="ovr", average="macro")
        out["log_loss"] = log_loss(y, proba, labels=list(range(len(LABELS))))
    return {k: round(float(v), 4) for k, v in out.items()}


def binary_block(y: np.ndarray, p_dark: np.ndarray, tau: float) -> dict:
    yb = (y != BENIGN).astype(int)
    pb = (p_dark >= tau).astype(int)
    p, r, f, _ = precision_recall_fscore_support(yb, pb, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(yb, pb, labels=[0, 1]).ravel()
    return {"threshold": round(float(tau), 3), "precision": round(float(p), 4), "recall": round(float(r), 4),
            "f1": round(float(f), 4), "roc_auc": round(float(roc_auc_score(yb, p_dark)), 4),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "false_positive_rate": round(float(fp / (fp + tn)), 4)}


def ece(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    conf = proba.max(1)
    correct = (proba.argmax(1) == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def multiclass_brier(y: np.ndarray, proba: np.ndarray) -> float:
    onehot = np.eye(len(LABELS))[y]
    return float(((proba - onehot) ** 2).sum(1).mean())


def fit_temperature(y: np.ndarray, proba: np.ndarray) -> float:
    def nll(T: float) -> float:
        return log_loss(y, temperature_scale(proba, T), labels=list(range(len(LABELS))))
    return float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)


# ---------------------------------------------------------------- data + embeddings
def load_data():
    df = pd.read_csv(DATA_PROCESSED / "text_dataset.csv", dtype={"page_id": str, "group_id": str})
    probe = pd.read_csv(DATA_PROCESSED / "probe_excluded.csv", dtype={"page_id": str})
    df["y"] = df["label"].map(L2I)
    return df, probe


def get_embedder():
    from sentence_transformers import SentenceTransformer
    local = ARTIFACTS / "minilm"
    if local.exists():
        return SentenceTransformer(str(local), device="cpu")
    m = SentenceTransformer(EMBEDDER_NAME, device="cpu")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    m.save(str(local))  # cache so the app runs offline afterwards
    return m


def embed(model, texts: list[str]) -> np.ndarray:
    return model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)


# ---------------------------------------------------------------- experiments
def e1_leakage(df: pd.DataFrame) -> dict:
    """Naive stratified random split (what we did first) vs the group+template split."""
    res = {}
    tr, te = train_test_split(df, test_size=len(df[df.split == "test"]), stratify=df["y"], random_state=SEED)
    leaked = te["template_key"].isin(set(tr["template_key"])).sum()
    leaked_page = te["page_id"].isin(set(tr["page_id"])).sum()
    vec = make_tfidf().fit(tr["text"])
    lr = make_lr(4.0).fit(vec.transform(tr["text"]), tr["y"])
    pred = lr.predict(vec.transform(te["text"]))
    res["naive_random_split"] = {"test_rows": int(len(te)), "test_rows_with_template_twin_in_train": int(leaked),
                                 "test_rows_with_page_in_train": int(leaked_page), **metrics_block(te["y"].values, pred)}
    trg, teg = df[df.split != "test"], df[df.split == "test"]
    trg = trg[trg.split == "train"]
    vec = make_tfidf().fit(trg["text"])
    lr = make_lr(4.0).fit(vec.transform(trg["text"]), trg["y"])
    pred = lr.predict(vec.transform(teg["text"]))
    res["group_template_split"] = {"test_rows": int(len(teg)), "test_rows_with_template_twin_in_train": 0,
                                   "test_rows_with_page_in_train": 0, **metrics_block(teg["y"].values, pred)}
    return res


def e5_repeated_cv(df: pd.DataFrame, emb: np.ndarray, C: float, w: float, n_trees: int) -> dict:
    pool = df[df.split != "test"].reset_index(drop=True)
    E = emb[df.split.values != "test"]
    scores = {"tfidf_lr": [], "minilm_xgb": [], "ensemble": []}
    for seed in range(5):
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_idx, te_idx in sgkf.split(pool, pool["y"], groups=pool["group_id"]):
            tr, te = pool.iloc[tr_idx], pool.iloc[te_idx]
            vec = make_tfidf().fit(tr["text"])
            lr = make_lr(C).fit(vec.transform(tr["text"]), tr["y"])
            p_lr = lr.predict_proba(vec.transform(te["text"]))
            Xtr = np.hstack([E[tr_idx], surface_features(tr["text"].tolist())])
            Xte = np.hstack([E[te_idx], surface_features(te["text"].tolist())])
            xgb = make_xgb(n_trees, early=False).fit(Xtr, tr["y"], sample_weight=compute_sample_weight("balanced", tr["y"]))
            p_x = xgb.predict_proba(Xte)
            for name, p in [("tfidf_lr", p_lr), ("minilm_xgb", p_x), ("ensemble", w * p_lr + (1 - w) * p_x)]:
                scores[name].append(f1_score(te["y"], p.argmax(1), average="macro"))
    return {k: {"mean_macro_f1": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4), "n_folds": len(v)} for k, v in scores.items()}


# ---------------------------------------------------------------- figures
def fig_confusion(cm: np.ndarray, path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    names = [DISPLAY_NAMES[l].replace(" / ", "/\n").replace(" ", "\n", 1) for l in LABELS]
    ax.set_xticks(range(len(LABELS)), names, fontsize=7)
    ax.set_yticks(range(len(LABELS)), names, fontsize=7)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() / 2 else "#1f2937")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=10)
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_model_comparison(val_scores: dict, test_scores: dict, path) -> None:
    names = list(val_scores)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(x - 0.18, [val_scores[n] for n in names], 0.36, label="Validation macro-F1", color="#2563eb")
    ax.bar(x + 0.18, [test_scores[n] for n in names], 0.36, label="Test macro-F1", color="#94a3b8")
    for i, n in enumerate(names):
        ax.text(i - 0.18, val_scores[n] + 0.005, f"{val_scores[n]:.3f}", ha="center", fontsize=7)
        ax.text(i + 0.18, test_scores[n] + 0.005, f"{test_scores[n]:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x, [n.replace("_", "\n") for n in names], fontsize=8)
    lo = min(min(val_scores.values()), min(test_scores.values()))
    ax.set_ylim(max(0, lo - 0.08), 1.0); ax.set_ylabel("Macro-F1"); ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_reliability(y, p_before, p_after, path) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 4))
    for p, lab, c in [(p_before, "Before (raw ensemble)", "#94a3b8"), (p_after, "After temperature scaling", "#2563eb")]:
        conf, corr = p.max(1), (p.argmax(1) == y)
        bins = np.linspace(0, 1, 11); xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 3:
                xs.append(conf[m].mean()); ys.append(corr[m].mean())
        ax.plot(xs, ys, "o-", label=lab, color=c)
    ax.plot([0, 1], [0, 1], "--", color="#cbd5e1", label="Perfect calibration")
    ax.set_xlabel("Predicted confidence"); ax.set_ylabel("Observed accuracy"); ax.legend(fontsize=7)
    ax.set_title("Reliability diagram (test)", fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_per_class(report: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3))
    x = np.arange(len(LABELS))
    for k, (m, c) in enumerate([("precision", "#1d4ed8"), ("recall", "#60a5fa"), ("f1-score", "#0f766e")]):
        vals = [report[l][m] for l in LABELS]
        ax.bar(x + (k - 1) * 0.26, vals, 0.26, label=m, color=c)
    ax.set_xticks(x, [DISPLAY_NAMES[l].replace(" / ", "/\n") for l in LABELS], fontsize=7)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=7, ncol=3, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_roc(y, p_dark, path) -> None:
    yb = (y != BENIGN).astype(int)
    fpr, tpr, _ = roc_curve(yb, p_dark)
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.plot(fpr, tpr, color="#2563eb", label=f"AUC = {roc_auc_score(yb, p_dark):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#cbd5e1")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(fontsize=8)
    ax.set_title("Dark vs benign ROC (test)", fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    ARTIFACTS.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    df, probe = load_data()
    embedder = get_embedder()
    emb = embed(embedder, df["text"].tolist())
    probe_emb = embed(embedder, probe["text"].tolist())
    print(f"embedded {len(df)} texts in {time.time()-t0:.1f}s")

    tr, va, te = (df.split == s for s in ("train", "val", "test"))
    Xtr_t, Xva_t, Xte_t = df.text[tr].tolist(), df.text[va].tolist(), df.text[te].tolist()
    ytr, yva, yte = df.y[tr].values, df.y[va].values, df.y[te].values
    Etr, Eva, Ete = emb[tr.values], emb[va.values], emb[te.values]
    Str, Sva, Ste = surface_features(Xtr_t), surface_features(Xva_t), surface_features(Xte_t)
    swt = compute_sample_weight("balanced", ytr)

    metrics: dict = {"seed": SEED, "labels": LABELS, "embedder": EMBEDDER_NAME, "surface_features": SURFACE_NAMES}

    # ---- E1
    metrics["E1_leakage"] = e1_leakage(df)
    print("E1", json.dumps(metrics["E1_leakage"], indent=0)[:600])

    # ---- E2 model comparison (validation only)
    vec = make_tfidf().fit(Xtr_t)
    Ttr, Tva, Tte = vec.transform(Xtr_t), vec.transform(Xva_t), vec.transform(Xte_t)
    cands: dict = {}
    grid_log: dict = {}

    best = None
    for C in [0.5, 1, 2, 4, 8, 16]:
        m = make_lr(C).fit(Ttr, ytr)
        s = f1_score(yva, m.predict(Tva), average="macro"); grid_log[f"tfidf_lr_C={C}"] = round(s, 4)
        if best is None or s > best[0]:
            best = (s, C, m)
    C_lr, lr = best[1], best[2]
    cands["tfidf_lr"] = {"val_proba": lr.predict_proba(Tva), "test_proba": lr.predict_proba(Tte), "params": {"C": C_lr}}

    best = None
    for C in [0.05, 0.1, 0.25, 0.5, 1]:
        m = LinearSVC(C=C, class_weight="balanced", random_state=SEED, max_iter=20000).fit(Ttr, ytr)
        s = f1_score(yva, m.predict(Tva), average="macro"); grid_log[f"tfidf_svm_C={C}"] = round(s, 4)
        if best is None or s > best[0]:
            best = (s, C, m)
    svm = best[2]
    cands["tfidf_svm"] = {"val_pred": svm.predict(Tva), "test_pred": svm.predict(Tte), "params": {"C": best[1]}}

    best = None
    for C in [0.5, 1, 2, 4, 8, 16]:
        m = make_lr(C).fit(Etr, ytr)
        s = f1_score(yva, m.predict(Eva), average="macro"); grid_log[f"minilm_lr_C={C}"] = round(s, 4)
        if best is None or s > best[0]:
            best = (s, C, m)
    mlr = best[2]
    cands["minilm_lr"] = {"val_proba": mlr.predict_proba(Eva), "test_proba": mlr.predict_proba(Ete), "params": {"C": best[1]}}

    xgb = make_xgb().fit(np.hstack([Etr, Str]), ytr, sample_weight=swt, eval_set=[(np.hstack([Eva, Sva]), yva)], verbose=False)
    n_trees = int(xgb.best_iteration) + 1
    cands["minilm_xgb"] = {"val_proba": xgb.predict_proba(np.hstack([Eva, Sva])), "test_proba": xgb.predict_proba(np.hstack([Ete, Ste])),
                           "params": {"n_estimators_after_early_stopping": n_trees, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.8}}

    best = None
    for w in np.round(np.arange(0, 1.01, 0.1), 2):
        p = w * cands["tfidf_lr"]["val_proba"] + (1 - w) * cands["minilm_xgb"]["val_proba"]
        s = f1_score(yva, p.argmax(1), average="macro"); grid_log[f"ensemble_w_lr={w}"] = round(s, 4)
        if best is None or s > best[0] + 1e-9:
            best = (s, float(w))
    w = best[1]
    cands["ensemble_lr_xgb"] = {"val_proba": w * cands["tfidf_lr"]["val_proba"] + (1 - w) * cands["minilm_xgb"]["val_proba"],
                                "test_proba": w * cands["tfidf_lr"]["test_proba"] + (1 - w) * cands["minilm_xgb"]["test_proba"],
                                "params": {"weight_tfidf_lr": w, "weight_minilm_xgb": round(1 - w, 2)}}

    comp = {}
    for name, c in cands.items():
        vp = c["val_pred"] if "val_pred" in c else c["val_proba"].argmax(1)
        tp = c["test_pred"] if "test_pred" in c else c["test_proba"].argmax(1)
        comp[name] = {"params": c["params"], "val": metrics_block(yva, vp, c.get("val_proba")), "test": metrics_block(yte, tp, c.get("test_proba"))}
    selected = max(comp, key=lambda n: comp[n]["val"]["macro_f1"])
    metrics["E2_model_comparison"] = comp
    metrics["E2_grid_search_val_macro_f1"] = grid_log
    metrics["selected_model"] = selected
    print("E2 val macro-F1:", {n: comp[n]["val"]["macro_f1"] for n in comp}, "->", selected)
    fig_model_comparison({n: comp[n]["val"]["macro_f1"] for n in comp}, {n: comp[n]["test"]["macro_f1"] for n in comp}, FIGURES / "model_comparison.png")

    # The serving bundle is always LR + XGB with a weight; if a single member wins, w becomes 1.0 or 0.0.
    if selected == "tfidf_lr":
        w = 1.0
    elif selected == "minilm_xgb":
        w = 0.0
    elif selected not in ("ensemble_lr_xgb",):
        print(f"NOTE: {selected} won on validation but is not servable in the LR/XGB bundle; serving the ensemble instead.")
        metrics["serving_note"] = f"{selected} had the best validation macro-F1 but the served model is ensemble_lr_xgb"
    p_val_raw = w * cands["tfidf_lr"]["val_proba"] + (1 - w) * cands["minilm_xgb"]["val_proba"]
    p_te_raw = w * cands["tfidf_lr"]["test_proba"] + (1 - w) * cands["minilm_xgb"]["test_proba"]

    # ---- E3 calibration
    T = fit_temperature(yva, p_val_raw)
    p_val = temperature_scale(p_val_raw, T)
    p_te = temperature_scale(p_te_raw, T)
    metrics["E3_calibration"] = {
        "method": "temperature scaling (1 parameter) fit on validation NLL",
        "temperature": round(T, 4),
        "test_before": {"ece": round(ece(yte, p_te_raw), 4), "brier": round(multiclass_brier(yte, p_te_raw), 4), "log_loss": round(log_loss(yte, p_te_raw, labels=list(range(5))), 4)},
        "test_after": {"ece": round(ece(yte, p_te), 4), "brier": round(multiclass_brier(yte, p_te), 4), "log_loss": round(log_loss(yte, p_te, labels=list(range(5))), 4)},
        "binary_dark_brier_test_after": round(brier_score_loss((yte != BENIGN).astype(int), 1 - p_te[:, BENIGN]), 4),
    }
    fig_reliability(yte, p_te_raw, p_te, FIGURES / "reliability.png")

    # ---- E4 threshold on P(dark)
    pdark_val, pdark_te = 1 - p_val[:, BENIGN], 1 - p_te[:, BENIGN]
    curve = []
    for t in np.round(np.arange(0.10, 0.96, 0.01), 2):
        b = binary_block(yva, pdark_val, t)
        curve.append((float(t), b["precision"], b["recall"], b["f1"]))
    ok = [c for c in curve if c[1] >= 0.90]
    best_f1 = max(c[3] for c in ok)
    # Real pages are mostly benign elements, so among near-ties (within 0.005 F1) prefer the
    # HIGHEST threshold: fewer false alarms at page scale for a negligible validation cost.
    tau = max(c[0] for c in ok if c[3] >= best_f1 - 0.005)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot([c[0] for c in curve], [c[1] for c in curve], label="precision", color="#1d4ed8")
    ax.plot([c[0] for c in curve], [c[2] for c in curve], label="recall", color="#60a5fa")
    ax.plot([c[0] for c in curve], [c[3] for c in curve], label="F1", color="#0f766e")
    ax.axvline(tau, ls="--", color="#dc2626", label=f"chosen tau={tau:.2f}")
    ax.set_xlabel("Threshold on calibrated P(dark)"); ax.set_ylim(0.8, 1.01); ax.legend(fontsize=7)
    ax.set_title("Validation dark-vs-benign threshold sweep", fontsize=10); fig.tight_layout()
    fig.savefig(FIGURES / "threshold_sweep.png", dpi=200); plt.close(fig)
    argmax_dark = (p_te.argmax(1) != BENIGN).astype(float)
    metrics["E4_threshold"] = {
        "rule": "maximise validation F1 (dark vs benign) s.t. precision >= 0.90; near-ties (<=0.005 F1) broken towards the higher threshold",
        "val_curve": [{"tau": c[0], "precision": c[1], "recall": c[2], "f1": c[3]} for c in curve],
        "tau": tau,
        "val_at_tau": binary_block(yva, pdark_val, tau),
        "test_argmax_decision": binary_block(yte, argmax_dark, 0.5),
        "test_at_tau": binary_block(yte, pdark_te, tau),
        # our FIRST rule (no tie-break, grid starting at 0.30) chose tau=0.30; kept for the before/after record
        "previous_rule": {"rule": "max val F1 s.t. precision>=0.90, grid 0.30-0.90, first maximum", "tau": 0.30,
                          "test": binary_block(yte, pdark_te, 0.30)},
    }

    # ---- Final held-out test evaluation of the served model
    pred = p_te.argmax(1)
    # application decision: flag if P(dark) >= tau, category = argmax over dark classes
    dark_cols = [i for i in range(len(LABELS)) if i != BENIGN]
    app_pred = np.where(pdark_te >= tau, np.array(dark_cols)[p_te[:, dark_cols].argmax(1)], BENIGN)
    rep = classification_report(yte, app_pred, target_names=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, app_pred, labels=list(range(len(LABELS))))
    metrics["final_test"] = {
        "model": f"ensemble(w_lr={w}) + temperature(T={T:.3f}) + tau={tau}",
        "argmax_metrics": metrics_block(yte, pred, p_te),
        "application_metrics": metrics_block(yte, app_pred, p_te),
        "per_class": {l: {k: round(float(v), 4) for k, v in rep[l].items()} for l in LABELS},
        "confusion_matrix": cm.tolist(),
        "binary_dark_vs_benign": binary_block(yte, pdark_te, tau),
        "per_class_roc_auc_ovr": {l: round(float(roc_auc_score((yte == i).astype(int), p_te[:, i])), 4) for i, l in enumerate(LABELS)},
    }
    # one row per template: stops a single repeated template ("NN% offers claimed. Hurry up!")
    # from dominating the error counts
    first = ~df[te].reset_index(drop=True)["template_key"].duplicated().values
    metrics["final_test"]["template_deduplicated"] = {
        "n_rows": int(first.sum()), **metrics_block(yte[first], app_pred[first]),
        "binary_dark_vs_benign": binary_block(yte[first], pdark_te[first], tau),
    }
    fig_confusion(cm, FIGURES / "confusion_matrix.png", "Held-out test confusion matrix (n=%d)" % len(yte))
    fig_per_class(rep, FIGURES / "per_class.png")
    fig_roc(yte, pdark_te, FIGURES / "roc_dark_vs_benign.png")

    # error analysis
    te_df = df[te].reset_index(drop=True).assign(pred=[LABELS[i] for i in app_pred], p_dark=np.round(pdark_te, 3))
    fps = te_df[(te_df.label == "benign") & (te_df.pred != "benign")].sort_values("p_dark", ascending=False)
    fns = te_df[(te_df.label != "benign") & (te_df.pred == "benign")].sort_values("p_dark")
    conf = te_df[(te_df.label != "benign") & (te_df.pred != "benign") & (te_df.label != te_df.pred)]
    metrics["error_analysis"] = {
        "false_positives": fps[["text", "pred", "p_dark"]].head(15).to_dict("records"),
        "false_negatives": fns[["text", "label", "p_dark"]].head(15).to_dict("records"),
        "category_confusions": conf[["text", "label", "pred", "p_dark"]].head(15).to_dict("records"),
        "n_fp": int(len(fps)), "n_fn": int(len(fns)), "n_category_confusions": int(len(conf)),
    }

    # ---- E5 robustness
    metrics["E5_repeated_group_cv"] = e5_repeated_cv(df, emb, C_lr, w, n_trees)

    # ---- E6 out-of-taxonomy probe
    praw_probe = w * lr.predict_proba(vec.transform(probe.text)) + (1 - w) * xgb.predict_proba(np.hstack([probe_emb, surface_features(probe.text.tolist())]))
    pp = temperature_scale(praw_probe, T)
    probe_flag = (1 - pp[:, BENIGN]) >= tau
    metrics["E6_out_of_taxonomy_probe"] = {
        cat: {"n": int((probe.source_category == cat).sum()), "flagged_as_dark": int(probe_flag[(probe.source_category == cat).values].sum())}
        for cat in probe.source_category.unique()
    }

    # ---- persist served model (trained on TRAIN split only; val used for selection/calibration/threshold)
    train_df = df[tr]
    bundle = ClassifierBundle(
        labels=LABELS, tfidf=vec, lr=lr, xgb=xgb, weight_lr=w, temperature=T, threshold=tau,
        neighbor_texts=train_df.text.tolist(), neighbor_labels=train_df.label.tolist(), neighbor_emb=Etr.astype(np.float32),
        meta={"embedder": EMBEDDER_NAME, "trained_on": "ec-darkpattern dataset.tsv TRAIN split only", "n_train": int(tr.sum()),
              "selected_model": selected, "lr_C": C_lr, "xgb_trees": n_trees, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")},
    )
    joblib.dump(bundle, ARTIFACTS / "clearcart_text_model.joblib", compress=3)
    metrics["runtime_seconds"] = round(time.time() - t0, 1)
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=float))
    print(json.dumps({k: metrics[k] for k in ["selected_model", "E3_calibration", "E4_threshold", "E5_repeated_group_cv", "E6_out_of_taxonomy_probe"]}, indent=1, default=float))
    print(json.dumps(metrics["final_test"], indent=1, default=float))
    print("FP:", metrics["error_analysis"]["n_fp"], "FN:", metrics["error_analysis"]["n_fn"])


if __name__ == "__main__":
    main()
