"""Local zero-shot classifier via NLI entailment.

The comparison has to be fair to be worth anything. A hosted decision model is
handed a label space it has never been trained on, so the local contender must
be too - a classifier fine-tuned on the task would win on accuracy while
answering a different question.

Natural language inference gives the local analogue of a typed Choice: score
"this text is about {label}" as a hypothesis against the text as premise, for
every label, and softmax the entailment logits into a distribution over labels.
That produces the same shape of answer - a label plus calibrated-ish
probabilities - from a model small enough to run on a CPU.
"""

from __future__ import annotations

import time

from ..types import Example, Prediction, TaskSpec

DEFAULT_MODEL = "typeform/distilbert-base-uncased-mnli"
HYPOTHESIS = "This text is about {}."


class HFLocalBackend:
    """Zero-shot NLI classification on local hardware."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        name: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 256,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "local backends need extra deps: pip install 'edgefront[local]'"
            ) from exc

        self._torch = torch
        self.name = name or f"{model_name.split('/')[-1]}-fp32"
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()
        self._model.to(device)

        # MNLI heads order their labels inconsistently across checkpoints, so
        # find the entailment index rather than assuming index 2.
        id2label = {
            int(k): str(v).lower() for k, v in self._model.config.id2label.items()
        }
        self._entail_idx = next(
            (i for i, lab in id2label.items() if "entail" in lab), len(id2label) - 1
        )
        self._hypotheses: dict[str, list[str]] = {}

    def _hypotheses_for(self, task: TaskSpec) -> list[str]:
        if task.name not in self._hypotheses:
            self._hypotheses[task.name] = [
                HYPOTHESIS.format(label.replace("_", " ")) for label in task.labels
            ]
        return self._hypotheses[task.name]

    def predict(self, ex: Example, task: TaskSpec) -> Prediction:
        torch = self._torch
        labels = task.labels
        hypotheses = self._hypotheses_for(task)

        start = time.perf_counter()
        scores: list[float] = []
        try:
            with torch.inference_mode():
                for i in range(0, len(hypotheses), self._batch_size):
                    chunk = hypotheses[i : i + self._batch_size]
                    encoded = self._tokenizer(
                        [ex.text] * len(chunk),
                        chunk,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self._max_length,
                        padding=True,
                    ).to(self._device)
                    logits = self._model(**encoded).logits
                    scores.extend(logits[:, self._entail_idx].tolist())
        except Exception as exc:
            return Prediction(
                label="",
                latency_ms=(time.perf_counter() - start) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )
        elapsed = (time.perf_counter() - start) * 1000

        tensor = torch.tensor(scores)
        probs = torch.softmax(tensor, dim=0).tolist()
        distribution = dict(zip(labels, probs, strict=True))
        best = max(distribution, key=lambda k: distribution[k])
        return Prediction(
            label=best,
            latency_ms=elapsed,
            probabilities=distribution,
            confidence=distribution[best],
            est_input_tokens=len(self._tokenizer.encode(ex.text)),
        )

    def meta(self) -> dict:
        params = sum(p.numel() for p in self._model.parameters())
        return {
            "kind": "local",
            "model": self._model_name,
            "precision": "fp32",
            "device": self._device,
            "offline": True,
            "params_m": round(params / 1e6, 1),
            "model_size_mb": round(params * 4 / 1e6, 1),
            "method": "zero-shot NLI entailment",
            "note": (
                "scores one hypothesis per label, so cost grows with the size "
                "of the label space"
            ),
        }
