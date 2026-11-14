"""Repeated grouped cross-validation for ALL candidate models (train+val pool only; test untouched).

Single validation splits are noisy at this data size (328 rows): the validation ranking and the
test ranking disagreed in our first run (SVM best on test, ensemble best on val). This script
measures each candidate over 5 seeds x 5 grouped folds = 25 fits to check whether the model
selection was a fluke. It does not modify the served model.

Run: python -m ml.robustness
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

import ml  # noqa: F401
from ml.common import ARTIFACTS, DATA_PROCESSED, LABELS, SEED
from ml.model import surface_features
from ml.train import embed, get_embedder, make_lr, make_tfidf, make_xgb


def main() -> None:
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    comp = metrics["E2_model_comparison"]
    C_lr = comp["tfidf_lr"]["params"]["C"]; C_svm = comp["tfidf_svm"]["params"]["C"]; C_mlr = comp["minilm_lr"]["params"]["C"]
    n_trees = comp["minilm_xgb"]["params"]["n_estimators_after_early_stopping"]
    w = comp["ensemble_lr_xgb"]["params"]["weight_tfidf_lr"]

    df = pd.read_csv(DATA_PROCESSED / "text_dataset.csv", dtype={"page_id": str, "group_id": str})
    pool = df[df.split != "test"].reset_index(drop=True)
    y = pool.label.map({l: i for i, l in enumerate(LABELS)}).values
    E = embed(get_embedder(), pool.text.tolist())
    S = surface_features(pool.text.tolist())
    names = ["tfidf_lr", "tfidf_svm", "minilm_lr", "minilm_xgb", "ensemble_lr_xgb"]
    scores = {n: [] for n in names}
    for seed in range(5):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED + seed).split(pool, y, pool.group_id):
            vec = make_tfidf().fit(pool.text.iloc[tr])
            Ttr, Tte = vec.transform(pool.text.iloc[tr]), vec.transform(pool.text.iloc[te])
            p_lr = make_lr(C_lr).fit(Ttr, y[tr]).predict_proba(Tte)
            svm = LinearSVC(C=C_svm, class_weight="balanced", random_state=SEED, max_iter=20000).fit(Ttr, y[tr])
            p_mlr = make_lr(C_mlr).fit(E[tr], y[tr]).predict_proba(E[te])
            xgb = make_xgb(n_trees, early=False).fit(np.hstack([E[tr], S[tr]]), y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
            p_x = xgb.predict_proba(np.hstack([E[te], S[te]]))
            preds = {"tfidf_lr": p_lr.argmax(1), "tfidf_svm": svm.predict(Tte), "minilm_lr": p_mlr.argmax(1),
                     "minilm_xgb": p_x.argmax(1), "ensemble_lr_xgb": (w * p_lr + (1 - w) * p_x).argmax(1)}
            for n in names:
                scores[n].append(float(f1_score(y[te], preds[n], average="macro")))
    out = {n: {"mean_macro_f1": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4),
               "min": round(float(np.min(v)), 4), "max": round(float(np.max(v)), 4), "folds": [round(s, 4) for s in v]}
           for n, v in scores.items()}
    wins = {n: 0 for n in names}
    for k in range(len(scores[names[0]])):
        wins[max(names, key=lambda n: scores[n][k])] += 1
    result = {"protocol": "5 seeds x StratifiedGroupKFold(5) on train+val pool (groups = page+template); hyper-parameters fixed to the values selected in ml/train.py",
              "n_pool": int(len(pool)), "results": out, "fold_wins": wins}
    (ARTIFACTS / "robustness.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({n: (out[n]["mean_macro_f1"], out[n]["std"]) for n in names}, indent=1), wins)


if __name__ == "__main__":
    main()
