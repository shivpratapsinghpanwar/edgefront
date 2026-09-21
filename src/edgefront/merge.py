"""Combine result documents produced on different machines.

The intended use: the local arm of a comparison runs on a GPU (Kaggle, say,
where there is no TypeSafe API key), and the hosted arm runs wherever the key
lives. Each run produces a normal `bench` result document; this recombines them
into one document with a single frontier and verdict, exactly as if every
backend had run in the same process.

Combining is only valid when the documents describe the same task on the same
examples - otherwise a "45 points" gap could just mean two different tests. So
this checks task name and example count, and refuses to merge a backend name
present in more than one document rather than silently picking one.
"""

from __future__ import annotations

from .bench import SCHEMA_VERSION
from .frontier import DEFAULT_TOLERANCE_PTS, decide, pareto
from .types import BackendResult


class MergeError(ValueError):
    pass


def _as_backend_result(entry: dict) -> BackendResult:
    return BackendResult(
        backend=entry["backend"],
        meta=entry.get("meta") or {},
        predictions=[],
        accuracy=entry.get("accuracy"),
        macro_f1=entry.get("macro_f1"),
        ece=entry.get("ece"),
        latency=entry.get("latency_ms") or {},
        cost=entry.get("cost") or {},
        n_errors=entry.get("n_errors", 0),
    )


def merge_documents(
    docs: list[dict], tolerance_pts: float = DEFAULT_TOLERANCE_PTS
) -> dict:
    """Merge `bench` result documents into one, recomputing frontier and verdict."""
    if not docs:
        raise MergeError("no documents to merge")
    if len(docs) == 1:
        return docs[0]

    task_names = {d["task"]["name"] for d in docs}
    if len(task_names) > 1:
        raise MergeError(f"documents describe different tasks: {sorted(task_names)}")

    example_counts = {d["task"]["n_examples"] for d in docs}
    if len(example_counts) > 1:
        raise MergeError(
            f"documents ran different example counts: {sorted(example_counts)} - "
            f"a fair comparison needs the same examples in every arm"
        )

    seen: dict[str, str] = {}  # backend name -> which document's environment
    entries: list[dict] = []
    environments = []
    for doc in docs:
        env_desc = (
            f"{doc.get('environment', {}).get('platform', 'unknown')} @ "
            f"{doc.get('environment', {}).get('utc_time', 'unknown')}"
        )
        for r in doc.get("results", []):
            name = r["backend"]
            if name in seen:
                raise MergeError(
                    f"backend '{name}' appears in more than one document "
                    f"({seen[name]} and {env_desc}) - merge would silently "
                    f"drop one of them"
                )
            seen[name] = env_desc
            entries.append(r)
        environments.append(doc.get("environment", {}))

    backend_results = [_as_backend_result(e) for e in entries]
    verdict = decide(backend_results, tolerance_pts=tolerance_pts)
    frontier = pareto(backend_results)

    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "bench-merged",
        "success": any(r.accuracy is not None for r in backend_results),
        "duration_s": sum(d.get("duration_s", 0.0) for d in docs),
        "task": docs[0]["task"],
        "environments": environments,
        "config": docs[0].get("config", {}),
        "results": entries,
        "frontier": frontier,
        "verdict": verdict.to_dict(),
        "error": None,
        "merged_from": len(docs),
    }
