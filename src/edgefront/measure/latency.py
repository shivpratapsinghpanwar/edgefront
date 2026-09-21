"""Latency statistics.

Two rules that decide whether a benchmark is honest:

* discard warmup calls - the first request pays for connection setup, lazy
  imports and cold caches, and including it flatters nothing but noise;
* report the distribution, never a bare mean - a mean hides the tail, and the
  tail is what breaks a real-time budget.
"""

from __future__ import annotations

DEFAULT_WARMUP = 3


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted list.

    Implemented here rather than pulled from numpy so that ``edgefront report``
    and ``edgefront verify`` run on a stdlib-only install.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def summarise(
    values: list[float], warmup: int = DEFAULT_WARMUP
) -> dict[str, float | int]:
    """Drop ``warmup`` leading samples, then describe what is left."""
    kept = values[warmup:] if len(values) > warmup else values[:]
    if not kept:
        return {"count": 0, "warmup_dropped": 0}
    ordered = sorted(kept)
    return {
        "count": len(ordered),
        "warmup_dropped": len(values) - len(kept),
        "p50": round(percentile(ordered, 0.50), 3),
        "p90": round(percentile(ordered, 0.90), 3),
        "p99": round(percentile(ordered, 0.99), 3),
        "max": round(ordered[-1], 3),
        "mean": round(sum(ordered) / len(ordered), 3),
    }
