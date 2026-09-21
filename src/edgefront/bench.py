"""Run a task across backends and assemble the result document.

Deliberately boring and sequential: backends must not run concurrently or they
would contend for the same CPU and corrupt each other's latency numbers.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import time
from dataclasses import dataclass

from .measure.accuracy import expected_calibration_error, score
from .measure.cost import HOSTED_PRICES_USD_PER_MTOK, CostModel, hosted_cost, local_cost
from .measure.latency import DEFAULT_WARMUP, summarise
from .types import Backend, BackendResult, TaskSpec

SCHEMA_VERSION = 1


@dataclass
class BenchConfig:
    warmup: int = DEFAULT_WARMUP
    limit: int | None = None
    cost_model: CostModel = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.cost_model is None:
            self.cost_model = CostModel()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def environment() -> dict:
    """What a reader needs to judge whether these numbers transfer to them."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "machine": platform.machine(),
        "git_commit": _git_commit(),
        "utc_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def run_backend(
    backend: Backend, task: TaskSpec, config: BenchConfig
) -> BackendResult:
    examples = task.examples[: config.limit] if config.limit else task.examples
    predictions = [backend.predict(ex, task) for ex in examples]

    meta = backend.meta()
    latencies = [p.latency_ms for p in predictions if p.ok]
    latency = summarise(latencies, warmup=config.warmup)

    accuracy, macro_f1 = score(predictions, examples)
    ece, curve = expected_calibration_error(predictions, examples)

    n_ok = sum(1 for p in predictions if p.ok)
    total_tokens = sum(p.est_input_tokens or 0 for p in predictions if p.ok)
    if meta.get("offline", False):
        cost = local_cost(float(latency.get("p50", 0.0)), config.cost_model)
    else:
        price = HOSTED_PRICES_USD_PER_MTOK.get(
            meta.get("vendor", ""), HOSTED_PRICES_USD_PER_MTOK["jev"]
        )
        cost = hosted_cost(total_tokens, max(1, n_ok), price)

    result = BackendResult(
        backend=backend.name,
        meta=meta,
        predictions=predictions,
        accuracy=accuracy,
        macro_f1=macro_f1,
        ece=ece,
        latency=latency,
        cost=cost.to_dict(),
        n_errors=sum(1 for p in predictions if not p.ok),
    )
    result.meta = {**meta, "reliability_curve": curve}
    return result


def build_document(
    task: TaskSpec,
    results: list[BackendResult],
    config: BenchConfig,
    duration_s: float,
    verdict_dict: dict,
    frontier_names: list[str],
) -> dict:
    """The result JSON. An agent or a CI job reads this, never the raw logs."""
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "bench",
        "success": any(r.accuracy is not None for r in results),
        "duration_s": round(duration_s, 3),
        "task": task.to_dict(),
        "environment": environment(),
        "config": {
            "warmup": config.warmup,
            "limit": config.limit,
            "cost_model": config.cost_model.to_dict(),
        },
        "results": [r.to_dict() for r in results],
        "frontier": frontier_names,
        "verdict": verdict_dict,
        "error": None,
    }
