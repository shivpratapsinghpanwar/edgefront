"""Local zero-shot classifier on ONNX Runtime, at whatever precision you built.

Same NLI scoring as the torch backend, so the only variable between them is the
runtime and the precision - which is the entire point.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from ..types import Example, Prediction, TaskSpec

HYPOTHESIS = "This text is about {}."


def _entailment_index(model_name: str) -> int | None:
    """Which logit column is 'entailment' for this checkpoint.

    MNLI heads do not agree on label order across checkpoints, so this is read
    from the config instead of assumed. Returns None when the config cannot be
    reached, and the caller falls back to the last column.
    """
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name)
        for idx, label in config.id2label.items():
            if "entail" in str(label).lower():
                return int(idx)
    except Exception:
        return None
    return None


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


class ONNXLocalBackend:
    """Zero-shot NLI classification through ONNX Runtime."""

    def __init__(
        self,
        model_path: str | Path,
        tokenizer_name: str,
        *,
        name: str | None = None,
        precision: str = "fp32",
        providers: list[str] | None = None,
        max_length: int = 256,
        entail_idx: int | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "local backends need extra deps: pip install 'edgefront[local]'"
            ) from exc

        self._path = Path(model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._max_length = max_length
        if providers is None:
            # get_available_providers() can front a remote provider, which
            # would quietly turn a "local" measurement into a network call.
            # Prefer GPU, then plain CPU, and never anything else by default.
            available = ort.get_available_providers()
            providers = [
                p
                for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                if p in available
            ] or ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(self._path), providers=providers)
        # Never let a caller believe they are on GPU when they are not: report
        # the provider the session actually bound, not the one that was asked
        # for. Silent CPU fallback is the classic benchmark lie.
        self._providers = self._session.get_providers()
        self._inputs = {i.name for i in self._session.get_inputs()}
        self._precision = precision
        self.name = name or f"{self._path.stem}-{precision}"
        self._hypotheses: dict[str, list[str]] = {}

        # Read the entailment index off the checkpoint config rather than
        # assuming the MNLI ordering. Guessing here is silently wrong rather
        # than loudly wrong: you get a plausible-looking accuracy computed from
        # the wrong logit, which is the worst failure a benchmark can have.
        self._entail_idx = entail_idx
        if self._entail_idx is None:
            self._entail_idx = _entailment_index(tokenizer_name)

    def _resolve_entail_idx(self, n_classes: int) -> int:
        if self._entail_idx is None or self._entail_idx >= n_classes:
            self._entail_idx = n_classes - 1
        return self._entail_idx

    def _hypotheses_for(self, task: TaskSpec) -> list[str]:
        if task.name not in self._hypotheses:
            self._hypotheses[task.name] = [
                HYPOTHESIS.format(label.replace("_", " ")) for label in task.labels
            ]
        return self._hypotheses[task.name]

    def predict(self, ex: Example, task: TaskSpec) -> Prediction:
        hypotheses = self._hypotheses_for(task)
        labels = task.labels

        start = time.perf_counter()
        try:
            encoded = self._tokenizer(
                [ex.text] * len(hypotheses),
                hypotheses,
                return_tensors="np",
                truncation=True,
                max_length=self._max_length,
                padding=True,
            )
            feed = {
                name: encoded[name]
                for name in ("input_ids", "attention_mask")
                if name in self._inputs
            }
            logits = self._session.run(None, feed)[0]
            idx = self._resolve_entail_idx(logits.shape[-1])
            scores = [float(row[idx]) for row in logits]
        except Exception as exc:
            return Prediction(
                label="",
                latency_ms=(time.perf_counter() - start) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )
        elapsed = (time.perf_counter() - start) * 1000

        distribution = dict(zip(labels, _softmax(scores), strict=True))
        best = max(distribution, key=lambda k: distribution[k])
        return Prediction(
            label=best,
            latency_ms=elapsed,
            probabilities=distribution,
            confidence=distribution[best],
            est_input_tokens=len(self._tokenizer.encode(ex.text)),
        )

    def meta(self) -> dict:
        return {
            "kind": "local",
            "runtime": "onnxruntime",
            "model_file": self._path.name,
            "precision": self._precision,
            "providers": self._providers,
            "offline": True,
            "model_size_mb": round(self._path.stat().st_size / 1e6, 2),
            "method": "zero-shot NLI entailment",
        }
