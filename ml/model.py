"""ClearCart text classifier: shared by training (ml/train.py) and serving (backend/).

Final architecture (selected on validation macro-F1, see ml/train.py):

    text --+--> TF-IDF (word 1-2g + char_wb 2-5g) --> Logistic Regression --> p_lr ---+
           |                                                                         +--> w*p_lr + (1-w)*p_xgb
           +--> all-MiniLM-L6-v2 (frozen, 384-d) + 7 surface feats --> XGBoost --> p_xgb
                                                                                     |
                                    temperature scaling (fit on validation) <--------+
                                                    |
                         calibrated P(class), P(dark) = 1 - P(benign), threshold tau

The MiniLM encoder is used purely as a frozen, general-purpose pretrained feature extractor.
It is NOT fine-tuned. All task-specific learning happens in LR / XGBoost on the permitted data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SURFACE_NAMES = [
    "log_chars", "n_words", "upper_ratio", "has_digit", "n_exclaim", "has_currency", "has_clock",
]
_CURRENCY = re.compile(r"[$£€¥₹]|\b(usd|eur|gbp|inr)\b", re.I)
_CLOCK = re.compile(r"\d{1,2}\s*:\s*\d{2}")


def surface_features(texts: list[str]) -> np.ndarray:
    """Generic surface statistics (Berkeley detector also mixes text signals with embeddings).

    These are numeric descriptors, not keyword rules: none of them decides a label on its own.
    """
    rows = []
    for t in texts:
        letters = [c for c in t if c.isalpha()]
        upper = sum(c.isupper() for c in letters) / len(letters) if letters else 0.0
        rows.append([
            np.log1p(len(t)),
            len(t.split()),
            upper,
            float(any(c.isdigit() for c in t)),
            t.count("!"),
            float(bool(_CURRENCY.search(t))),
            float(bool(_CLOCK.search(t))),
        ])
    return np.asarray(rows, dtype=np.float32)


# Abstention gate (added after a real-world false positive): a string with no words - only prices,
# numbers and symbols, e.g. "₹ 2,990.00 ₹ 3,950.00" - gives the text model no linguistic evidence.
# The trained model scores such strings as urgency (it learned "digits + ':' / '.' = countdown"), so
# the application abstains instead of guessing. Measured on the held-out test split by
# ml/eval_abstention.py (ml/artifacts/abstention_eval.json).
_WORD = re.compile(r"[^\W\d_]{2,}")
_NON_WORDS = {"rs", "inr", "usd", "eur", "gbp", "aud", "cad", "jpy", "cny"}


def has_lexical_evidence(text: str) -> bool:
    return any(w.lower() not in _NON_WORDS for w in _WORD.findall(text or ""))


# ---------------------------------------------------------------------------------------------------------
# False-positive gates (post-hoc, application level). They never change predict() or its probabilities; the
# Auditor uses them to decide whether a score that crosses the threshold becomes a FINDING. Measured by
# ml/eval_false_positives.py (ml/artifacts/false_positive_eval.json).
#
# Gate 1 - quote-aware scoring ("mention vs use"). Explanatory text that QUOTES an example of a dark pattern
# ("'Only 2 left!' or 'selling fast' claims that cannot be verified.") is not itself using it. Rule:
#   * quoted span = text between an opening quote (" “ ‘ ') and the first closing quote (" ” ’ '), at most 200
#     characters; an apostrophe inside a word (don't / don’t) never opens or closes a span; open/close
#     characters may differ (OCR confuses them); an opening quote with no closing quote that runs to the end of
#     the element also counts (OCR lines and wrapped text are cut mid-quotation). The last two clauses were
#     added after the About-page screenshot probe, whose OCR text had mismatched and cut-off quotes;
#   * the rule APPLIES only if the text has >= 1 quoted span AND >= MIN_UNQUOTED_WORDS words (>= 2 letters)
#     left after masking. A text that is entirely a quotation, or has little context around the quote, is
#     scored normally - '"No thanks, I hate saving money"' on a button is a use, not a mention;
#   * if it applies, every quoted span (quotes included) is replaced by QUOTE_PLACEHOLDER and the masked text
#     is scored. If the masked text is below the threshold, the element is NOT a finding: it is reported as
#     "quoted example - treated as a mention, not a claim" (never dropped silently). If the masked text is
#     still above the threshold, the element stays a finding with its ORIGINAL score.
#   QUOTE_PLACEHOLDER was chosen on the VALIDATION split as the most score-neutral of 9 candidates: appending
#   it to each of the 328 val texts moved P(dark) by 0.0011 on average and flipped no decision.
QUOTE_PLACEHOLDER = "…"
MIN_UNQUOTED_WORDS = 4
_W = r"[^\W_]"   # a letter or digit
_OPEN, _CLOSE, _ANYQ = "\"“‘'", "\"”’'", "\"“”‘’'"
QUOTE_SPAN = re.compile(
    # closed span: an opening quote not glued to a preceding letter/digit, then the FIRST valid closing quote
    # (one not followed by a letter/digit, so the apostrophe in don't / don’t never closes a span).
    # Opening and closing characters may differ ('...’ , ‘...") because OCR routinely confuses straight
    # and curly quotes.
    rf"(?<!{_W})[{_OPEN}](?=\S)[^\n]{{1,200}}?(?<=\S)[{_CLOSE}](?!{_W})"
    # unclosed span running to the END of the element (an OCR line or HTML text node cut mid-quotation):
    # opening quote + a letter, then no further quote marks except in-word apostrophes, up to the end.
    rf"|(?<!{_W})[{_OPEN}](?=[^\W\d_])(?:[^{_ANYQ}\n]|(?<={_W})['’](?={_W})){{1,200}}$"
)


def quoted_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in QUOTE_SPAN.finditer(text or "")]


def quote_masked(text: str) -> str | None:
    """The masked text if the mention-vs-use rule applies to `text`, else None (score the text normally)."""
    if not text or not QUOTE_SPAN.search(text):
        return None
    masked = re.sub(r"\s+", " ", QUOTE_SPAN.sub(QUOTE_PLACEHOLDER, text)).strip()
    if len(_WORD.findall(masked)) < MIN_UNQUOTED_WORDS:
        return None
    return masked


# Gate 2 - out-of-distribution abstention. Temperature scaling was fit on validation data, so a calibrated
# "99%" only means something for text that resembles the training data. similarity = max cosine similarity
# between the element's embedding and the TRAIN-split neighbour bank (clf.b.neighbor_emb, the 1,645 train
# texts). Threshold rule (VALIDATION split only): tau_ood = the k-th smallest similarity among validation
# DARK rows plus one position, k = floor(OOD_VAL_DARK_QUANTILE * n_dark), i.e. the largest value such that at
# most 2% of validation dark rows fall strictly below it. Elements below tau_ood are not findings; they are
# reported as "uncertain - unlike the training data".
OOD_VAL_DARK_QUANTILE = 0.02


def max_train_similarity(bundle, emb: np.ndarray) -> np.ndarray | None:
    """Max cosine similarity of each embedding (rows of `emb`, L2-normalised) to the training neighbour bank."""
    bank = getattr(bundle, "neighbor_emb", None)
    if bank is None or not len(bank):
        return None
    emb = np.atleast_2d(np.asarray(emb, dtype=np.float32))
    return (emb @ np.asarray(bank, dtype=np.float32).T).max(axis=1)


def ood_threshold_from_val(clf, quantile: float = OOD_VAL_DARK_QUANTILE) -> dict | None:
    """Apply the tau_ood rule to the validation split with `clf`'s own embedding space. None if unavailable."""
    import pandas as pd

    from ml.common import DATA_PROCESSED
    path = DATA_PROCESSED / "text_dataset.csv"
    if getattr(clf.b, "neighbor_emb", None) is None or not path.exists():
        return None
    df = pd.read_csv(path, keep_default_na=False)
    va = df[(df.split == "val") & (df.label != "benign")]
    _, emb = clf.raw_proba(va.text.tolist())
    sims = np.sort(max_train_similarity(clf.b, emb))
    k = int(np.floor(quantile * len(sims)))
    tau = float(sims[k])
    return {"threshold": tau, "rule": f"largest value with <= {quantile:.0%} of validation dark rows strictly below it "
                                      f"({k} of {len(sims)} rows; tau = sorted similarity #{k + 1})",
            "quantile": quantile, "n_val_dark": int(len(sims)), "n_val_dark_below": int((sims < tau).sum())}


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def temperature_scale(p: np.ndarray, T: float) -> np.ndarray:
    return softmax(np.log(np.clip(p, 1e-9, 1.0)) / T)


def build_predictions(texts: list[str], cal: np.ndarray, emb: np.ndarray, labels: list[str], threshold: float) -> list[dict]:
    """The predict() contract shared by every ClearCart text model (calibrated probabilities in -> dicts out)."""
    benign_idx = labels.index("benign")
    out = []
    for i, t in enumerate(texts):
        p = cal[i]
        p_dark = float(1.0 - p[benign_idx])
        dark_probs = {lab: float(p[j]) for j, lab in enumerate(labels) if lab != "benign"}
        top_cat = max(dark_probs, key=dark_probs.get)
        abstained = not has_lexical_evidence(t)
        out.append({
            "text": t,
            "probabilities": {lab: float(p[j]) for j, lab in enumerate(labels)},
            "p_dark": p_dark,
            "category": top_cat,
            # share of the dark-pattern mass that belongs to the top category
            "category_share": float(dark_probs[top_cat] / p_dark) if p_dark > 0 else 0.0,
            "flagged": (p_dark >= threshold) and not abstained,
            "abstained": abstained,          # no words -> no linguistic evidence -> never flagged
            "_emb": emb[i],
        })
    return out


@dataclass
class ClassifierBundle:
    labels: list[str]
    tfidf: object
    lr: object
    xgb: object
    weight_lr: float
    temperature: float
    threshold: float
    neighbor_texts: list[str] = field(default_factory=list)
    neighbor_labels: list[str] = field(default_factory=list)
    neighbor_emb: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


class TextClassifier:
    """Inference wrapper. Load once, call many times."""

    def __init__(self, bundle: ClassifierBundle, embedder) -> None:
        self.b = bundle
        self.embedder = embedder
        self.labels = bundle.labels
        self.benign_idx = self.labels.index("benign")
        vocab = bundle.tfidf.get_feature_names_out()
        self._word_mask = np.array([n.startswith("word__") for n in vocab])
        self._vocab = np.array([n.split("__", 1)[1] for n in vocab])

    @classmethod
    def load(cls, artifacts_dir: Path) -> "TextClassifier":
        return cls.load_file(artifacts_dir / "clearcart_text_model.joblib", artifacts_dir / "minilm")

    @classmethod
    def load_file(cls, bundle_path: Path, embedder_dir: Path) -> "TextClassifier":
        """Load a ClassifierBundle from any path (used for ml/artifacts_v2 bundles, which reuse the frozen MiniLM)."""
        import joblib
        from sentence_transformers import SentenceTransformer

        bundle: ClassifierBundle = joblib.load(bundle_path)
        embedder = SentenceTransformer(str(embedder_dir) if embedder_dir.exists() else bundle.meta["embedder"], device="cpu")
        return cls(bundle, embedder)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.embedder.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)

    def raw_proba(self, texts: list[str], emb: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        if emb is None:
            emb = self.embed(texts)
        p_lr = self.b.lr.predict_proba(self.b.tfidf.transform(texts))
        p_xgb = self.b.xgb.predict_proba(np.hstack([emb, surface_features(texts)]))
        return self.b.weight_lr * p_lr + (1 - self.b.weight_lr) * p_xgb, emb

    def predict(self, texts: list[str]) -> list[dict]:
        if not texts:
            return []
        raw, emb = self.raw_proba(texts)
        cal = temperature_scale(raw, self.b.temperature)
        return build_predictions(texts, cal, emb, self.labels, self.b.threshold)

    # ---------- explainability helpers (all derived from the actual input) ----------
    def key_terms(self, text: str, label: str, k: int = 4) -> list[str]:
        """Words/phrases in `text` that pushed the TF-IDF+LR component towards `label`."""
        x = self.b.tfidf.transform([text])
        # LR was fit on integer ids 0..K-1 in LABELS order
        cls_idx = list(self.b.lr.classes_).index(self.labels.index(label))
        coef = self.b.lr.coef_[cls_idx]
        idx = x.indices
        contrib = x.data * coef[idx]
        order = np.argsort(-contrib)
        terms = []
        for j in order:
            fi = idx[j]
            if contrib[j] <= 0 or not self._word_mask[fi]:
                continue
            term = self._vocab[fi]
            if term.lower() in text.lower() and term not in terms:
                terms.append(term)
            if len(terms) >= k:
                break
        return terms

    def neighbors(self, emb: np.ndarray, k: int = 2) -> list[dict]:
        """Most similar labelled TRAINING examples (cosine similarity in MiniLM space)."""
        if self.b.neighbor_emb is None:
            return []
        sims = self.b.neighbor_emb @ emb
        top = np.argsort(-sims)[:k]
        return [{"text": self.b.neighbor_texts[i], "label": self.b.neighbor_labels[i], "similarity": round(float(sims[i]), 3)} for i in top]


# ================================================================ v2: fine-tuned transformer (opt-in)
@dataclass
class TransformerBundle:
    """Bundle-like metadata so callers that read `clf.b` (backend versions stamp, Auditor threshold) keep working."""
    labels: list[str]
    temperature: float
    threshold: float
    meta: dict = field(default_factory=dict)
    weight_lr: float | None = None            # not an LR/XGB ensemble
    neighbor_texts: list[str] = field(default_factory=list)
    neighbor_labels: list[str] = field(default_factory=list)
    neighbor_emb: np.ndarray | None = None


_OCC_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)?")


class TransformerClassifier:
    """Fine-tuned distilroberta-base, same predict() contract as TextClassifier.

    * probabilities: softmax(logits), then the same temperature_scale() used for the ensemble (T fit on VAL).
    * _emb: L2-normalised mean-pooled last hidden state of the fine-tuned encoder (used by neighbors()).
    * key_terms(): leave-one-word-out occlusion on the transformer itself - a word is a key term if deleting it
      lowers the calibrated probability of `label`. Model-faithful, needs no second model.
    * Runs on CPU at inference so served scores are deterministic and identical to ml/artifacts_v2/metrics.json.
    """
    kind = "transformer"

    def __init__(self, model, tokenizer, bundle: TransformerBundle, max_length: int = 128, batch_size: int = 32) -> None:
        self.model, self.tok, self.b = model, tokenizer, bundle
        self.labels = bundle.labels
        self.benign_idx = self.labels.index("benign")
        self.max_length, self.batch_size = max_length, batch_size

    @classmethod
    def load(cls, model_dir: Path) -> "TransformerClassifier":
        import json

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        cfg = json.loads((model_dir / "clearcart_config.json").read_text())
        tok = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to("cpu").eval()
        nb = np.load(model_dir / "neighbors.npz", allow_pickle=False)
        bundle = TransformerBundle(labels=cfg["labels"], temperature=float(cfg["temperature"]), threshold=float(cfg["threshold"]),
                                   meta=cfg.get("meta", {}), neighbor_texts=nb["texts"].tolist(),
                                   neighbor_labels=nb["labels"].tolist(), neighbor_emb=nb["emb"].astype(np.float32))
        return cls(model, tok, bundle, max_length=int(cfg.get("max_length", 128)))

    def forward(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """(logits [n,K], normalised pooled embeddings [n,H])."""
        import torch
        logits, embs = [], []
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tok(texts[i:i + self.batch_size], truncation=True, max_length=self.max_length, padding=True, return_tensors="pt")
                out = self.model(**enc, output_hidden_states=True)
                h = out.hidden_states[-1]
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
                logits.append(out.logits.float().numpy()); embs.append(pooled.float().numpy())
        return np.vstack(logits), np.vstack(embs)

    def raw_proba(self, texts: list[str], emb: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        logits, e = self.forward(texts)
        return softmax(logits), e

    def predict(self, texts: list[str]) -> list[dict]:
        if not texts:
            return []
        raw, emb = self.raw_proba(texts)
        cal = temperature_scale(raw, self.b.temperature)
        return build_predictions(texts, cal, emb, self.labels, self.b.threshold)

    def key_terms(self, text: str, label: str, k: int = 4, max_words: int = 48) -> list[str]:
        words = []
        for w in _OCC_WORD.findall(text):
            if any(c.isalpha() for c in w) and w.lower() not in {x.lower() for x in words}:
                words.append(w)
            if len(words) >= max_words:
                break
        if not words:
            return []
        variants = [re.sub(r"(?<![^\W_])" + re.escape(w) + r"(?![^\W_])", " ", text, flags=re.I) for w in words]
        variants = [re.sub(r"\s+", " ", v).strip() for v in variants]
        raw, _ = self.raw_proba([text] + variants)
        cal = temperature_scale(raw, self.b.temperature)
        j = self.labels.index(label)
        drop = cal[0, j] - cal[1:, j]
        order = np.argsort(-drop)
        return [words[i] for i in order if drop[i] > 0][:k]

    def neighbors(self, emb: np.ndarray, k: int = 2) -> list[dict]:
        """Most similar labelled TRAINING examples (cosine similarity of the fine-tuned encoder's pooled embedding)."""
        if self.b.neighbor_emb is None or not len(self.b.neighbor_texts):
            return []
        sims = self.b.neighbor_emb @ emb
        top = np.argsort(-sims)[:k]
        return [{"text": self.b.neighbor_texts[i], "label": self.b.neighbor_labels[i], "similarity": round(float(sims[i]), 3)} for i in top]


def load_classifier(name: str | None = None):
    """Model selection for serving.

    name None / "" / "v1" / "default" -> the frozen served model in ml/artifacts (unchanged default).
    "v2" -> the VAL-selected winner recorded in ml/artifacts_v2/selected.json.
    "ensemble_v2" | "transformer_v2" | "transformer_v1data" -> that specific v2 artefact.
    """
    import json

    from ml.common import ARTIFACTS, ROOT
    v2 = ROOT / "ml" / "artifacts_v2"
    name = (name or "").strip().lower()
    if name in ("", "v1", "default"):
        return TextClassifier.load(ARTIFACTS)
    if name == "v2":
        name = json.loads((v2 / "selected.json").read_text())["selected_model"]
    if name == "ensemble_v2":
        return TextClassifier.load_file(v2 / "ensemble_v2" / "clearcart_text_model.joblib", ARTIFACTS / "minilm")
    if name in ("transformer_v2", "transformer_v1data"):
        return TransformerClassifier.load(v2 / name)
    raise ValueError(f"unknown CLEARCART_MODEL '{name}' (expected v1, v2, ensemble_v2, transformer_v2, transformer_v1data)")
