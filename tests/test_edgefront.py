"""Offline test suite: no API key, no model download, no network."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from edgefront.backends.rules import RulesBackend
from edgefront.backends.stub import StubBackend
from edgefront.bench import BenchConfig, build_document, run_backend
from edgefront.frontier import decide, pareto
from edgefront.measure.accuracy import expected_calibration_error, score
from edgefront.measure.cost import CostModel, hosted_cost, local_cost
from edgefront.measure.latency import percentile, summarise
from edgefront.report import render_markdown
from edgefront.tasks import load_task
from edgefront.types import BackendResult, Example, Prediction


# ----------------------------------------------------------------- latency --
def test_percentile_interpolates():
    assert percentile([0.0, 10.0], 0.5) == 5.0
    assert percentile([1.0], 0.99) == 1.0
    assert percentile([], 0.5) == 0.0


def test_summarise_drops_warmup():
    # a slow first call must not pollute the distribution
    stats = summarise([100.0, 1.0, 1.0, 2.0, 2.0, 2.0], warmup=3)
    assert stats["count"] == 3
    assert stats["warmup_dropped"] == 3
    assert stats["max"] == 2.0


def test_summarise_keeps_everything_when_too_few():
    stats = summarise([5.0, 6.0], warmup=3)
    assert stats["count"] == 2
    assert stats["warmup_dropped"] == 0


# ---------------------------------------------------------------- accuracy --
def _ex(uid, gold):
    return Example(uid=uid, text="t", gold=gold)


def test_score_counts_errors_as_wrong():
    examples = [_ex("a", "x"), _ex("b", "y")]
    preds = [
        Prediction(label="x", latency_ms=1.0),
        Prediction(label="", latency_ms=1.0, error="boom"),
    ]
    accuracy, _ = score(preds, examples)
    assert accuracy == 0.5


def test_perfect_confident_predictions_have_zero_ece():
    examples = [_ex(str(i), "x") for i in range(10)]
    preds = [
        Prediction(label="x", latency_ms=1.0, confidence=1.0) for _ in examples
    ]
    ece, curve = expected_calibration_error(preds, examples)
    assert ece == 0.0
    assert sum(b["count"] for b in curve) == 10


def test_overconfident_and_always_wrong_is_maximally_miscalibrated():
    examples = [_ex(str(i), "x") for i in range(10)]
    preds = [
        Prediction(label="y", latency_ms=1.0, confidence=1.0) for _ in examples
    ]
    ece, _ = expected_calibration_error(preds, examples)
    assert ece == 1.0


def test_ece_is_none_without_confidence():
    examples = [_ex("a", "x")]
    preds = [Prediction(label="x", latency_ms=1.0)]
    ece, curve = expected_calibration_error(preds, examples)
    assert ece is None and curve == []


# -------------------------------------------------------------------- cost --
def test_hosted_cost_arithmetic():
    # 1000 tok/call at $0.042/Mtok = $0.000042/call = $42 per million calls
    report = hosted_cost(total_input_tokens=1000, n_calls=1, price_per_mtok=0.042)
    assert round(report.usd_per_million_calls, 2) == 42.0


def test_local_cost_falls_with_faster_inference():
    model = CostModel()
    fast = local_cost(10.0, model).usd_per_call
    slow = local_cost(100.0, model).usd_per_call
    assert fast < slow


def test_local_cost_records_its_assumptions():
    report = local_cost(10.0, CostModel(watts=99.0))
    assert report.assumptions["watts"] == 99.0
    assert "99" in report.basis


# ---------------------------------------------------------------- frontier --
def _result(name, acc, p50, offline, kind="local", ece=None, cost=0.0):
    return BackendResult(
        backend=name,
        meta={"offline": offline, "kind": kind},
        predictions=[],
        accuracy=acc,
        ece=ece,
        latency={"p50": p50},
        cost={"usd_per_million_calls": cost},
    )


def test_pareto_drops_dominated():
    results = [
        _result("fast_good", 0.9, 10, True),
        _result("slow_bad", 0.8, 50, True),
    ]
    assert pareto(results) == ["fast_good"]


def test_pareto_keeps_genuine_tradeoff():
    results = [
        _result("accurate_slow", 0.95, 100, False, kind="hosted"),
        _result("quick_rough", 0.85, 5, True),
    ]
    assert set(pareto(results)) == {"accurate_slow", "quick_rough"}


def test_verdict_recommends_local_when_close():
    results = [
        _result("jev", 0.91, 140, False, kind="hosted", cost=42.0),
        _result("local_int8", 0.89, 30, True, cost=1.1),
    ]
    verdict = decide(results)
    assert verdict.recommended == "local_int8"
    assert "within" in verdict.headline


def test_verdict_recommends_hosted_when_gap_is_real():
    results = [
        _result("jev", 0.95, 140, False, kind="hosted", cost=42.0),
        _result("local_int8", 0.70, 30, True, cost=1.1),
    ]
    verdict = decide(results)
    assert verdict.recommended == "jev"
    assert "does not substitute" in verdict.headline


def test_verdict_flags_a_task_the_baseline_nearly_solves():
    results = [
        _result("jev", 0.62, 140, False, kind="hosted"),
        _result("rules", 0.60, 1, True, kind="rules"),
    ]
    verdict = decide(results)
    assert any("too easy" in line for line in verdict.detail)


def test_verdict_notes_calibration_winner():
    results = [
        _result("jev", 0.91, 140, False, kind="hosted", ece=0.04, cost=42.0),
        _result("local_int8", 0.90, 30, True, ece=0.15, cost=1.1),
    ]
    verdict = decide(results)
    assert any("calibrated" in line for line in verdict.detail)


# ------------------------------------------------------------ end-to-end ----
def test_bench_runs_offline_and_report_renders():
    task = load_task("synthetic", n=40)
    config = BenchConfig(limit=40)
    results = [run_backend(b, task, config) for b in (RulesBackend(), StubBackend())]
    verdict = decide(results)
    doc = build_document(task, results, config, 1.0, verdict.to_dict(), pareto(results))

    assert doc["success"] is True
    assert doc["schema_version"] == 1
    assert len(doc["results"]) == 2
    # the document must survive a JSON round trip - CI reads it, not the logs
    assert json.loads(json.dumps(doc, sort_keys=True))["task"]["name"] == "synthetic"

    markdown = render_markdown(doc)
    assert "## Verdict" in markdown
    assert "warmup" in markdown


def test_stub_is_deterministic():
    task = load_task("synthetic", n=30)
    config = BenchConfig(limit=30)
    a = run_backend(StubBackend(), task, config)
    b = run_backend(StubBackend(), task, config)
    assert a.accuracy == b.accuracy
    assert [p.label for p in a.predictions] == [p.label for p in b.predictions]


def test_rules_beats_chance_on_synthetic():
    task = load_task("synthetic", n=120)
    result = run_backend(RulesBackend(), task, BenchConfig(limit=120))
    assert result.accuracy > 0.25  # 4 labels, so chance is 0.25


# ------------------------------------------------------- entailment index --
# `typeform/distilbert-base-uncased-mnli` orders its labels ENTAILMENT,
# NEUTRAL, CONTRADICTION - entailment is index 0, not the conventional 2 (the
# last column). Assuming n_classes - 1 silently scores the *contradiction*
# logit instead: accuracy still looks plausible, it is just answering the
# opposite question. See HANDOVER.md. This must never regress unnoticed, so
# these tests fake `transformers` in sys.modules and never touch the network
# or download a model.
def _install_fake_transformers(monkeypatch, id2label=None, raises=None):
    class FakeConfig:
        pass

    config = FakeConfig()
    if id2label is not None:
        config.id2label = id2label

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(name):
            if raises is not None:
                raise raises
            return config

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoConfig = FakeAutoConfig
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_entailment_index_resolved_from_checkpoint_config(monkeypatch):
    from edgefront.backends import onnx_local

    _install_fake_transformers(
        monkeypatch,
        id2label={0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"},
    )
    assert onnx_local._entailment_index("typeform/distilbert-base-uncased-mnli") == 0


def test_entailment_index_returns_none_when_config_unreachable(monkeypatch):
    from edgefront.backends import onnx_local

    _install_fake_transformers(monkeypatch, raises=OSError("no network"))
    assert onnx_local._entailment_index("unreachable/checkpoint") is None


def test_resolve_entail_idx_uses_the_configured_column():
    from edgefront.backends.onnx_local import ONNXLocalBackend

    backend = object.__new__(ONNXLocalBackend)
    backend._entail_idx = 0
    assert backend._resolve_entail_idx(n_classes=3) == 0


def test_resolve_entail_idx_falls_back_to_last_column_when_unknown():
    from edgefront.backends.onnx_local import ONNXLocalBackend

    backend = object.__new__(ONNXLocalBackend)
    backend._entail_idx = None
    assert backend._resolve_entail_idx(n_classes=3) == 2


# --------------------------------------------------------- cli exit codes ---
def _run_cli(*args, cwd):
    return subprocess.run(
        [sys.executable, "-m", "edgefront.cli", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def test_cli_bench_and_verify_exit_codes(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "bench.json"
    proc = _run_cli(
        "bench", "--task", "synthetic", "--backends", "rules,stub",
        "--limit", "40", "--json", str(out), cwd=root,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()

    ok = _run_cli("verify", str(out), "--min-acc", "0.2", cwd=root)
    assert ok.returncode == 0

    bad = _run_cli("verify", str(out), "--min-acc", "0.999", cwd=root)
    assert bad.returncode == 1
    assert "FAIL" in bad.stdout


def test_cli_rejects_unknown_backend(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    proc = _run_cli("bench", "--backends", "nope", cwd=root)
    assert proc.returncode == 2


# ------------------------------------------------------------------- merge --
from edgefront.merge import MergeError, merge_documents  # noqa: E402


def _doc(task_name, n_examples, backend_names, env_time="2026-01-01T00:00:00Z"):
    return {
        "task": {"name": task_name, "n_examples": n_examples, "labels": ["x", "y"]},
        "environment": {"platform": "test", "utc_time": env_time},
        "config": {"warmup": 3},
        "duration_s": 1.0,
        "results": [
            {
                "backend": name,
                "meta": {"offline": name != "jev"},
                "accuracy": 0.8,
                "ece": 0.1,
                "latency_ms": {"p50": 10.0},
                "cost": {"usd_per_million_calls": 1.0},
                "n_errors": 0,
            }
            for name in backend_names
        ],
    }


def test_merge_combines_disjoint_backends():
    a = _doc("synthetic", 40, ["rules", "onnx_int8"])
    b = _doc("synthetic", 40, ["jev"], env_time="2026-01-02T00:00:00Z")
    merged = merge_documents([a, b])
    names = {r["backend"] for r in merged["results"]}
    assert names == {"rules", "onnx_int8", "jev"}
    assert merged["merged_from"] == 2
    assert len(merged["environments"]) == 2


def test_merge_rejects_mismatched_task():
    a = _doc("synthetic", 40, ["rules"])
    b = _doc("banking77", 40, ["jev"])
    with pytest.raises(MergeError, match="different tasks"):
        merge_documents([a, b])


def test_merge_rejects_mismatched_example_count():
    a = _doc("synthetic", 40, ["rules"])
    b = _doc("synthetic", 60, ["jev"])
    with pytest.raises(MergeError, match="different example counts"):
        merge_documents([a, b])


def test_merge_rejects_duplicate_backend_name():
    a = _doc("synthetic", 40, ["rules"])
    b = _doc("synthetic", 40, ["rules"])
    with pytest.raises(MergeError, match="more than one document"):
        merge_documents([a, b])


def test_merge_single_document_is_passthrough():
    a = _doc("synthetic", 40, ["rules"])
    assert merge_documents([a]) is a


def test_merged_report_renders_multiple_environments():
    a = _doc("synthetic", 40, ["rules"])
    b = _doc("synthetic", 40, ["jev"], env_time="2026-01-02T00:00:00Z")
    merged = merge_documents([a, b])
    md = render_markdown(merged)
    assert "merges 2 runs from different machines" in md
    assert "2026-01-01T00:00:00Z" in md
    assert "2026-01-02T00:00:00Z" in md
