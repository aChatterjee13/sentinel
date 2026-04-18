"""Lightweight text statistics monitoring."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

_TOKEN_PATTERN = re.compile(r"\b[\w']+\b", re.UNICODE)


def tokenise(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_PATTERN.findall(text or "")]


@dataclass
class TextStats:
    """Aggregated text statistics for a corpus snapshot."""

    n_documents: int
    avg_length: float
    avg_token_count: float
    oov_rate: float
    unique_tokens: int
    new_tokens: list[str]


class TextStatsMonitor:
    """Maintain a running vocabulary and report drift signals."""

    def __init__(self, oov_threshold: float = 0.05, top_new_tokens_k: int = 50):
        self.oov_threshold = oov_threshold
        self.top_new_tokens_k = top_new_tokens_k
        self._vocab: set[str] = set()

    def fit(self, texts: Iterable[str]) -> None:
        self._vocab = set()
        for text in texts:
            self._vocab.update(tokenise(text))

    def evaluate(self, texts: Iterable[str]) -> TextStats:
        docs = list(texts)
        if not docs:
            return TextStats(0, 0.0, 0.0, 0.0, len(self._vocab), [])
        char_lengths = [len(d or "") for d in docs]
        token_lists = [tokenise(d) for d in docs]
        token_counts = [len(t) for t in token_lists]
        flat = [tok for sub in token_lists for tok in sub]
        counter = Counter(flat)
        unseen = {tok: cnt for tok, cnt in counter.items() if tok not in self._vocab}
        oov_rate = (
            (sum(unseen.values()) / sum(token_counts))
            if token_counts and sum(token_counts)
            else 0.0
        )
        new_tokens = [
            t
            for t, _ in sorted(unseen.items(), key=lambda kv: kv[1], reverse=True)[
                : self.top_new_tokens_k
            ]
        ]
        return TextStats(
            n_documents=len(docs),
            avg_length=float(sum(char_lengths) / len(docs)),
            avg_token_count=float(sum(token_counts) / len(docs)),
            oov_rate=float(oov_rate),
            unique_tokens=len(counter),
            new_tokens=new_tokens,
        )

    def expand_vocab(self, texts: Iterable[str]) -> int:
        before = len(self._vocab)
        for text in texts:
            self._vocab.update(tokenise(text))
        return len(self._vocab) - before
