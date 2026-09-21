"""Keyword baseline - the floor every other backend must clear.

This exists to keep the headline honest. If a hosted frontier model is only a
few points above keyword matching on your task, that is the most important
number in the report, and it is the one a benchmark without a baseline hides.
"""

from __future__ import annotations

import re
import time

from ..types import Example, Prediction, TaskSpec

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class RulesBackend:
    """Scores each label by overlap between the text and the label's criteria."""

    name = "rules"

    def __init__(self) -> None:
        self._vocab: dict[str, set[str]] = {}

    def _ensure(self, task: TaskSpec) -> None:
        if self._vocab:
            return
        for label, description in task.criteria.items():
            words = set(_tokens(description)) | set(_tokens(label))
            self._vocab[label] = {w for w in words if len(w) > 3}

    def predict(self, ex: Example, task: TaskSpec) -> Prediction:
        self._ensure(task)
        start = time.perf_counter()
        words = set(_tokens(ex.text))
        scores = {
            label: len(words & vocab) for label, vocab in self._vocab.items()
        }
        total = sum(scores.values())
        if total == 0:
            # No evidence either way. Say so with a flat distribution rather
            # than letting dict ordering masquerade as a decision.
            labels = task.labels
            probs = {label: 1 / len(labels) for label in labels}
            best = labels[0]
        else:
            probs = {label: s / total for label, s in scores.items()}
            best = max(scores, key=lambda k: scores[k])
        elapsed = (time.perf_counter() - start) * 1000
        return Prediction(
            label=best,
            latency_ms=elapsed,
            probabilities=probs,
            confidence=probs[best],
            est_input_tokens=len(words),
        )

    def meta(self) -> dict:
        return {
            "kind": "rules",
            "precision": "n/a",
            "offline": True,
            "model_size_mb": 0.0,
        }
