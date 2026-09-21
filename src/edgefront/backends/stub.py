"""A deterministic fake backend.

Exists so the entire pipeline - bench, frontier, report, verify - can be
exercised in CI with no API key, no model download and no network, and so the
committed fixtures never drift. It is also the reference for anyone adding a
new backend: this is the whole contract.
"""

from __future__ import annotations

import hashlib
import time

from ..types import Example, Prediction, TaskSpec


class StubBackend:
    name = "stub"

    def __init__(self, accuracy: float = 0.75, seed: int = 0) -> None:
        self._accuracy = accuracy
        self._seed = seed

    def predict(self, ex: Example, task: TaskSpec) -> Prediction:
        start = time.perf_counter()
        digest = hashlib.sha256(f"{self._seed}:{ex.uid}".encode()).digest()
        roll = digest[0] / 255.0
        labels = task.labels
        if roll < self._accuracy:
            label = ex.gold
        else:
            others = [x for x in labels if x != ex.gold] or labels
            label = others[digest[1] % len(others)]
        confidence = 0.5 + (digest[2] / 255.0) * 0.5
        probs = {x: (1 - confidence) / max(1, len(labels) - 1) for x in labels}
        probs[label] = confidence
        elapsed = (time.perf_counter() - start) * 1000
        return Prediction(
            label=label,
            latency_ms=elapsed,
            probabilities=probs,
            confidence=confidence,
            est_input_tokens=max(1, len(ex.text) // 4),
        )

    def meta(self) -> dict:
        return {
            "kind": "stub",
            "precision": "n/a",
            "offline": True,
            "model_size_mb": 0.0,
            "note": "deterministic fake for tests and CI; not a real model",
        }
