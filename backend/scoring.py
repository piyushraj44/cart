"""Request-scoped inference cache.

Within one audit/journey request the auditor, MCAI (semantic dimension), the counterfactual planner and the
optimiser all score text with the same deterministic classifier, often the same strings. `CachedScorer`
memoises predictions by exact text for the lifetime of ONE request (a ContextVar set by the endpoint), so no
text is classified twice. The cache is deliberately not shared across requests: that would keep users'
uploaded text in server memory, which the privacy-first design avoids.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

_CURRENT: ContextVar[Optional["CachedScorer"]] = ContextVar("clearcart_scorer", default=None)


class CachedScorer:
    def __init__(self, clf) -> None:
        self.clf = clf
        self.cache: dict[str, dict] = {}
        self.model_calls = 0          # batched classifier invocations
        self.texts_scored = 0         # distinct texts sent to the classifier
        self.requests = 0             # texts requested by callers (incl. cache hits)

    def predict(self, texts: list[str]) -> list[dict]:
        self.requests += len(texts)
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if missing:
            self.model_calls += 1
            self.texts_scored += len(missing)
            for t, p in zip(missing, self.clf.predict(missing)):
                self.cache[t] = p
        # shallow copies: callers set keys such as 'flagged' on the returned dicts
        return [dict(self.cache[t]) for t in texts]

    def stats(self) -> dict:
        return {"model_calls": self.model_calls, "texts_scored": self.texts_scored, "texts_requested": self.requests,
                "cache_hits": self.requests - self.texts_scored}


def current_scorer() -> Optional[CachedScorer]:
    return _CURRENT.get()


@contextmanager
def request_scorer(clf):
    token = _CURRENT.set(CachedScorer(clf))
    try:
        yield _CURRENT.get()
    finally:
        _CURRENT.reset(token)


def predict(clf, texts: list[str]) -> list[dict]:
    """Score through the request cache when one is active, else directly."""
    s = _CURRENT.get()
    return s.predict(texts) if s is not None else clf.predict(texts)
