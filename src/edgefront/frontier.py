"""Pareto frontier and the verdict.

The verdict line is the product; everything else in a run is evidence for it.
So it is computed from explicit, stated rules rather than prose, and it is
willing to say "the hosted model wins" - a tool that can only reach one
conclusion is not a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import BackendResult

# How close a local backend must be, in accuracy points, before we call it a
# practical substitute. 3 points is a judgement call and is printed with the
# verdict so a reader can disagree with it explicitly.
DEFAULT_TOLERANCE_PTS = 3.0


@dataclass(frozen=True)
class Verdict:
    headline: str
    detail: list[str]
    recommended: str | None
    tolerance_pts: float

    def to_dict(self) -> dict:
        return {
            "headline": self.headline,
            "detail": self.detail,
            "recommended": self.recommended,
            "tolerance_pts": self.tolerance_pts,
        }


def pareto(results: list[BackendResult]) -> list[str]:
    """Names on the accuracy/latency frontier.

    A backend is dominated when another is at least as accurate AND at least as
    fast, with one of the two strictly better.
    """
    keep: list[str] = []
    for a in results:
        if a.accuracy is None or not a.latency:
            continue
        a_p50 = float(a.latency.get("p50", 0.0))
        dominated = False
        for b in results:
            if b is a or b.accuracy is None or not b.latency:
                continue
            b_p50 = float(b.latency.get("p50", 0.0))
            at_least_as_good = b.accuracy >= a.accuracy and b_p50 <= a_p50
            strictly_better = b.accuracy > a.accuracy or b_p50 < a_p50
            if at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            keep.append(a.backend)
    return keep


def decide(
    results: list[BackendResult], tolerance_pts: float = DEFAULT_TOLERANCE_PTS
) -> Verdict:
    scored = [r for r in results if r.accuracy is not None]
    if not scored:
        return Verdict("no backend produced a usable result", [], None, tolerance_pts)

    hosted = [r for r in scored if not r.meta.get("offline", False)]
    local = [
        r
        for r in scored
        if r.meta.get("offline", False) and r.meta.get("kind") != "rules"
    ]
    baseline = next((r for r in scored if r.meta.get("kind") == "rules"), None)
    best = max(scored, key=lambda r: r.accuracy or 0.0)

    detail: list[str] = []

    # The check that keeps the whole exercise honest.
    if baseline is not None and baseline.accuracy is not None:
        gap = (best.accuracy - baseline.accuracy) * 100
        if gap < 10:
            detail.append(
                f"the keyword baseline is within {gap:.1f} pts of the best "
                f"backend - this task may be too easy to separate them"
            )

    if not hosted or not local:
        return Verdict(
            headline=f"{best.backend} leads at {best.accuracy:.1%} accuracy",
            detail=detail
            + ["no hosted/local pair was run, so no substitution verdict"],
            recommended=best.backend,
            tolerance_pts=tolerance_pts,
        )

    host = max(hosted, key=lambda r: r.accuracy or 0.0)
    best_local = max(local, key=lambda r: r.accuracy or 0.0)
    gap_pts = (host.accuracy - best_local.accuracy) * 100

    host_p50 = float(host.latency.get("p50", 0.0)) or 1e-9
    local_p50 = float(best_local.latency.get("p50", 0.0)) or 1e-9
    speedup = host_p50 / local_p50
    host_cost = float(host.cost.get("usd_per_million_calls", 0.0))
    local_cost = float(best_local.cost.get("usd_per_million_calls", 0.0)) or 1e-9
    cheaper = host_cost / local_cost if host_cost else 0.0

    if gap_pts <= tolerance_pts:
        headline = (
            f"{best_local.backend} is within {gap_pts:.1f} pts of "
            f"{host.backend} at {speedup:.1f}x lower p50"
        )
        if cheaper > 1:
            headline += f" and {cheaper:.0f}x lower cost"
        recommended = best_local.backend
    else:
        headline = (
            f"{host.backend} leads {best_local.backend} by {gap_pts:.1f} pts "
            f"- the local model does not substitute on this task"
        )
        recommended = host.backend

    if host.ece is not None and best_local.ece is not None:
        better = host if host.ece < best_local.ece else best_local
        detail.append(
            f"{better.backend} is better calibrated "
            f"(ECE {host.ece:.3f} hosted vs {best_local.ece:.3f} local)"
        )
        if recommended == best_local.backend and better is host:
            detail.append(
                "prefer the hosted model if you act on the probabilities "
                "rather than only the label"
            )

    if host.n_errors:
        detail.append(
            f"{host.backend} failed on {host.n_errors} call(s); those count as "
            f"wrong, not as missing"
        )

    return Verdict(headline, detail, recommended, tolerance_pts)
