"""Result JSON -> markdown.

A pure text transform: no numpy, no torch, no network. That is what lets the
README's numbers be regenerated in CI from the committed result file, so they
cannot quietly drift away from what was actually measured.
"""

from __future__ import annotations


def _fmt(value: object, spec: str = "", dash: str = "-") -> str:
    if value is None:
        return dash
    if spec:
        try:
            return format(value, spec)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _sig(value: object, dash: str = "-") -> str:
    """Format a magnitude without rounding it away.

    A sub-millisecond backend shown as "0.0" makes the table lie by rounding;
    so does a local cost of $0.004 per million calls shown as "0.00". Scale the
    precision to the value instead of picking one width for everything.
    """
    if value is None:
        return dash
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    if a >= 0.01:
        return f"{v:.3f}"
    return f"{v:.2e}"


def render_table(doc: dict) -> str:
    rows = []
    header = (
        f"{'backend':<20} {'acc':>6} {'ECE':>6} {'p50ms':>8} {'p99ms':>8} "
        f"{'$/1M':>9}  offline"
    )
    rows.append(header)
    rows.append("-" * len(header))
    for r in doc.get("results", []):
        latency = r.get("latency_ms") or {}
        cost = r.get("cost") or {}
        rows.append(
            f"{r['backend']:<20} "
            f"{_fmt(r.get('accuracy'), '.3f'):>6} "
            f"{_fmt(r.get('ece'), '.3f'):>6} "
            f"{_sig(latency.get('p50')):>8} "
            f"{_sig(latency.get('p99')):>8} "
            f"{_sig(cost.get('usd_per_million_calls')):>9}  "
            f"{'yes' if (r.get('meta') or {}).get('offline') else 'no'}"
        )
    return "\n".join(rows)


def render_markdown(doc: dict) -> str:
    task = doc.get("task", {})
    env = doc.get("environment", {})
    verdict = doc.get("verdict", {})
    out: list[str] = []

    out.append(f"# edgefront - {task.get('name', 'unknown')}")
    out.append("")
    out.append(
        f"{task.get('n_examples', 0)} examples, {len(task.get('labels', []))} labels. "
        f"{task.get('source', '')}"
    )
    out.append("")

    out.append("## Verdict")
    out.append("")
    out.append(f"**{verdict.get('headline', 'no verdict')}**")
    out.append("")
    for line in verdict.get("detail", []):
        out.append(f"- {line}")
    if verdict.get("recommended"):
        out.append(
            f"- recommended: **{verdict['recommended']}** "
            f"(substitution tolerance {verdict.get('tolerance_pts', 0):.1f} pts)"
        )
    out.append("")

    out.append("## Results")
    out.append("")
    out.append("| backend | accuracy | macro-F1 | ECE | p50 ms | p90 ms | p99 ms | "
               "$/1M calls | offline | errors |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|:--:|---:|")
    for r in doc.get("results", []):
        lat = r.get("latency_ms") or {}
        cost = r.get("cost") or {}
        meta = r.get("meta") or {}
        out.append(
            f"| {r['backend']} "
            f"| {_fmt(r.get('accuracy'), '.3f')} "
            f"| {_fmt(r.get('macro_f1'), '.3f')} "
            f"| {_fmt(r.get('ece'), '.3f')} "
            f"| {_sig(lat.get('p50'))} "
            f"| {_sig(lat.get('p90'))} "
            f"| {_sig(lat.get('p99'))} "
            f"| {_sig(cost.get('usd_per_million_calls'))} "
            f"| {'yes' if meta.get('offline') else 'no'} "
            f"| {r.get('n_errors', 0)} |"
        )
    out.append("")

    frontier = doc.get("frontier") or []
    if frontier:
        out.append(f"On the accuracy/latency frontier: {', '.join(frontier)}.")
        out.append("")

    out.append("## How these numbers were produced")
    out.append("")
    cfg = doc.get("config", {})
    out.append(
        f"- first {cfg.get('warmup', 0)} calls per backend discarded as warmup; "
        f"latency is the distribution over the rest, never a bare mean"
    )
    out.append("- backends run sequentially so they cannot contend for CPU")
    out.append("- a failed call counts as wrong, not as missing")
    for r in doc.get("results", []):
        basis = (r.get("cost") or {}).get("basis")
        if basis:
            out.append(f"- `{r['backend']}` cost basis: {basis}")
    out.append("")

    out.append("## Environment")
    out.append("")
    environments = doc.get("environments")
    if environments:
        out.append(
            f"This document merges {doc.get('merged_from', len(environments))} "
            f"runs from different machines - backends were NOT run sequentially "
            f"against each other, only within each source run:"
        )
        out.append("")
        for e in environments:
            out.append(
                f"- {e.get('platform', 'unknown')} / {e.get('processor', '')}, "
                f"python {e.get('python', '?')}, run {e.get('utc_time', '?')} UTC"
                + (f", commit `{e['git_commit']}`" if e.get("git_commit") else "")
            )
    else:
        out.append(f"- {env.get('platform', 'unknown')} / {env.get('processor', '')}")
        out.append(
            f"- python {env.get('python', '?')}, run {env.get('utc_time', '?')} UTC"
        )
        if env.get("git_commit"):
            out.append(f"- edgefront commit `{env['git_commit']}`")
    out.append(
        "- hosted latency depends on network position; rerun locally before "
        "trusting it for your own deployment"
    )
    out.append("")
    return "\n".join(out)
