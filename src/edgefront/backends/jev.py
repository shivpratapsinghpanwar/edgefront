"""The hosted-decision-model backend (TypeSafe Jev).

Every vendor-specific line in this project lives in this file. If the SDK, the
pricing or the vendor changes, nothing outside this module needs to move - the
tasks, quantization, measurement and reporting layers stay a working
cross-precision benchmark on their own.
"""

from __future__ import annotations

import os
import time

from ..types import Example, Prediction, TaskSpec

# Jev accepts at most 255 options in a single Choice.
MAX_CHOICE_OPTIONS = 255


class JevBackend:
    name = "jev"

    def __init__(self, model: str | None = None) -> None:
        if not os.environ.get("TYPESAFE_API_KEY"):
            raise RuntimeError(
                "TYPESAFE_API_KEY is not set. Get a key at typesafe.ai, or run "
                "with --backends rules,stub to exercise the pipeline offline."
            )
        try:
            from typesafe_sdk import Choice, TypeSafeClient
        except ImportError as exc:  # pragma: no cover - import-guard
            raise RuntimeError(
                "typesafe-sdk is not installed. pip install 'edgefront[jev]'"
            ) from exc
        self._Choice = Choice
        self._client = TypeSafeClient()
        self._model = model
        self._last_error: str | None = None

    def predict(self, ex: Example, task: TaskSpec) -> Prediction:
        if len(task.criteria) > MAX_CHOICE_OPTIONS:
            return Prediction(
                label="",
                latency_ms=0.0,
                error=(
                    f"task has {len(task.criteria)} labels; the Choice ceiling "
                    f"is {MAX_CHOICE_OPTIONS}"
                ),
            )
        question = self._Choice(
            instructions=task.instructions, criteria=dict(task.criteria)
        )
        state = {"text": ex.text}
        if ex.context:
            state.update(ex.context)

        start = time.perf_counter()
        try:
            response = self._client.system_one(
                state=state, questions={"label": question}
            )
        except Exception as exc:  # network, rate limit, auth
            return Prediction(
                label="",
                latency_ms=(time.perf_counter() - start) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )
        elapsed = (time.perf_counter() - start) * 1000

        answer = response.answers["label"]
        probs = getattr(answer, "probabilities", None)
        return Prediction(
            label=answer.choice,
            latency_ms=elapsed,
            probabilities=dict(probs) if probs else None,
            confidence=getattr(answer, "confidence", None),
            est_input_tokens=_estimate_tokens(state, task),
        )

    def meta(self) -> dict:
        return {
            "kind": "hosted",
            "vendor": "typesafe",
            "model": self._model or "jev-latest",
            "precision": "n/a",
            "offline": False,
            "note": (
                "hosted latency is dominated by network position; see "
                "'host' in the run environment block"
            ),
        }


def _estimate_tokens(state: dict, task: TaskSpec) -> int:
    """Rough input-token count for the cost model.

    Deliberately crude and documented as such: ~4 characters per token over the
    serialised state plus the question text. It is used only to price a call,
    and the alternative - shipping a tokenizer in the core install - would cost
    more than the precision is worth.
    """
    payload = " ".join(str(v) for v in state.values())
    payload += task.instructions
    payload += " ".join(f"{k} {v}" for k, v in task.criteria.items())
    return max(1, len(payload) // 4)
