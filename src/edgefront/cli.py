"""edgefront command line interface.

Follows the same contract as data-doctor: --json on every subcommand, and the
exit code is the verdict, so `edgefront verify` drops straight into CI.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .bench import BenchConfig, build_document, run_backend
from .frontier import DEFAULT_TOLERANCE_PTS, decide, pareto
from .measure.cost import CostModel
from .report import render_markdown, render_table
from .tasks import list_tasks, load_task
from .types import Backend

EXIT_OK = 0
EXIT_FAILED_CHECK = 1
EXIT_BAD_USAGE = 2


def _write_json(data: dict, out: str | None) -> None:
    if out:
        Path(out).write_text(
            json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nreport written to {out}")


def _headline(ok: bool, message: str) -> None:
    print(("  OK   " if ok else "  FAIL ") + message)


def _build_backend(spec: str) -> Backend:
    """Turn a --backends token into a backend instance.

    Import is lazy and per-backend so that a missing optional dependency only
    breaks the backend that needs it.
    """
    name = spec.strip().lower()
    if name == "rules":
        from .backends.rules import RulesBackend

        return RulesBackend()
    if name == "stub":
        from .backends.stub import StubBackend

        return StubBackend()
    if name == "jev":
        from .backends.jev import JevBackend

        return JevBackend()
    if name.startswith("hf:") or name == "hf":
        from .backends.hf_local import DEFAULT_MODEL, HFLocalBackend

        model = spec.split(":", 1)[1] if ":" in spec else DEFAULT_MODEL
        return HFLocalBackend(model)
    raise ValueError(
        f"unknown backend: {spec}. Known: rules, stub, jev, hf[:model-id]"
    )


def cmd_tasks(args: argparse.Namespace) -> int:
    for name, description in list_tasks().items():
        print(f"  {name:<12} {description}")
    return EXIT_OK


def cmd_bench(args: argparse.Namespace) -> int:
    try:
        task = load_task(args.task)
    except (ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_BAD_USAGE

    backends = []
    for spec in args.backends.split(","):
        if not spec.strip():
            continue
        try:
            backends.append(_build_backend(spec))
        except (ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_BAD_USAGE
    if not backends:
        print("error: no backends selected", file=sys.stderr)
        return EXIT_BAD_USAGE

    config = BenchConfig(
        warmup=args.warmup,
        limit=args.limit,
        cost_model=CostModel(
            hardware_usd=args.hardware_usd,
            watts=args.watts,
            utilisation=args.utilisation,
        ),
    )
    n = len(task.examples[: args.limit] if args.limit else task.examples)
    print(f"task {task.name}: {n} examples, {len(task.labels)} labels\n")

    started = time.perf_counter()
    results = []
    for backend in backends:
        print(f"running {backend.name} ...", flush=True)
        results.append(run_backend(backend, task, config))
    duration = time.perf_counter() - started

    verdict = decide(results, tolerance_pts=args.tolerance)
    frontier = pareto(results)
    doc = build_document(
        task, results, config, duration, verdict.to_dict(), frontier
    )

    print()
    print(render_table(doc))
    print()
    print(f"verdict: {verdict.headline}")
    for line in verdict.detail:
        print(f"         {line}")

    _write_json(doc, args.json)
    if args.md:
        Path(args.md).write_text(render_markdown(doc), encoding="utf-8")
        print(f"markdown written to {args.md}")
    return EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    doc = json.loads(Path(args.results).read_text(encoding="utf-8"))
    markdown = render_markdown(doc)
    if args.out:
        Path(args.out).write_text(markdown, encoding="utf-8")
        print(f"markdown written to {args.out}")
    else:
        print(markdown)
    return EXIT_OK


def cmd_verify(args: argparse.Namespace) -> int:
    doc = json.loads(Path(args.results).read_text(encoding="utf-8"))
    results = doc.get("results", [])
    if not results:
        print("  FAIL no results in document")
        return EXIT_FAILED_CHECK

    target = args.backend
    if target:
        results = [r for r in results if r["backend"] == target]
        if not results:
            print(f"  FAIL no backend named {target} in document")
            return EXIT_FAILED_CHECK

    failed = False
    for r in results:
        name = r["backend"]
        latency = r.get("latency_ms") or {}
        if args.min_acc is not None:
            acc = r.get("accuracy")
            ok = acc is not None and acc >= args.min_acc
            failed |= not ok
            _headline(ok, f"{name}: accuracy {acc:.3f} >= {args.min_acc:.3f}")
        if args.max_p99 is not None:
            p99 = latency.get("p99")
            ok = p99 is not None and p99 <= args.max_p99
            failed |= not ok
            _headline(ok, f"{name}: p99 {p99:.1f}ms <= {args.max_p99:.1f}ms")
        if args.max_ece is not None:
            ece = r.get("ece")
            ok = ece is not None and ece <= args.max_ece
            failed |= not ok
            _headline(ok, f"{name}: ECE {ece} <= {args.max_ece}")
        if r.get("n_errors"):
            failed = True
            _headline(False, f"{name}: {r['n_errors']} failed call(s)")

    return EXIT_FAILED_CHECK if failed else EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edgefront",
        description=(
            "Do you need a hosted decision model, or does a quantized local "
            "model match it? Measure it."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("tasks", help="list available decision tasks")
    p.set_defaults(func=cmd_tasks)

    p = sub.add_parser("bench", help="run a task across backends")
    p.add_argument("--task", default="synthetic")
    p.add_argument(
        "--backends",
        default="rules,stub",
        help="comma separated: rules, stub, jev, hf[:model-id]",
    )
    p.add_argument("--limit", type=int, default=None, help="cap examples")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_PTS)
    p.add_argument("--hardware-usd", type=float, default=600.0)
    p.add_argument("--watts", type=float, default=45.0)
    p.add_argument("--utilisation", type=float, default=0.25)
    p.add_argument("--json", default=None, help="write the result document here")
    p.add_argument("--md", default=None, help="write a markdown report here")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("report", help="render a result document as markdown")
    p.add_argument("results")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("verify", help="assert thresholds; exit 1 if any fail")
    p.add_argument("results")
    p.add_argument("--backend", default=None, help="check only this backend")
    p.add_argument("--min-acc", type=float, default=None)
    p.add_argument("--max-p99", type=float, default=None)
    p.add_argument("--max-ece", type=float, default=None)
    p.set_defaults(func=cmd_verify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
