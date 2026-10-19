from copy import deepcopy
import json
import subprocess
import sys

import pytest

from evaluation.comparison import (
    COMPATIBLE_FIELDS, METRICS, ComparisonError, compare_evaluations, render_report,
)


@pytest.fixture
def runs():
    metadata = dict.fromkeys(COMPATIBLE_FIELDS, "fixed")
    metadata.update(configured_sample_count=3, sample_ids=["s1", "s2", "s3"],
                    timestamp_utc="2026-09-20", run_mode="baseline")
    rows = [{"sample_id": sid, "question": "question", "reference_answer": "reference",
             "reference_context_ids": ["a"], "reference_contexts": ["a text"],
             "candidate_context_ids": ["a", "b", "c"], "actual_candidate_count": 3,
             "retrieved_context_ids": ["a", "b"], "retrieved_contexts": ["a text", "b text"],
             "metrics": dict.fromkeys(METRICS, 0.5), "metric_status": dict.fromkeys(METRICS, "success"),
             "errors": [], "status": "completed"} for sid in metadata["sample_ids"]]
    b = {"metadata": metadata, "samples": rows, "summary": {"original": True}}
    r = deepcopy(b)
    r["metadata"]["run_mode"] = "reranker"
    return b, r


@pytest.mark.parametrize("field", COMPATIBLE_FIELDS)
def test_metadata_mismatch_rejected(runs, field):
    b, r = runs
    r["metadata"][field] = "different"
    with pytest.raises(ComparisonError, match="metadata"):
        compare_evaluations(b, r)


def test_sample_set_mismatch_rejected(runs):
    b, r = runs
    r["samples"][0]["sample_id"] = "missing"
    r["metadata"]["sample_ids"][0] = "missing"
    with pytest.raises(ComparisonError, match="sets differ"):
        compare_evaluations(b, r)


def test_candidate_pool_mismatch_detected(runs):
    b, r = runs
    r["samples"][0]["candidate_context_ids"][-1] = "d"
    c = compare_evaluations(b, r)
    assert c["metadata_validation"]["candidate_pool_issues"][0]["sample_id"] == "s1"
    assert not c["samples"][0]["reranker_attribution_valid"]


def test_paired_intersection_uses_same_subset_and_discloses_failures(runs):
    b, r = runs
    name = METRICS[0]
    b["samples"][0]["metrics"][name] = 0.0
    r["samples"][0]["metrics"][name] = 0.2
    b["samples"][1]["metrics"][name] = None
    b["samples"][1]["metric_status"][name] = "failed"
    b["samples"][1]["errors"] = [{"stage": name, "error_type": "ValueError", "message": "safe"}]
    r["samples"][2]["metrics"][name] = None
    r["samples"][2]["metric_status"][name] = "skipped"
    c = compare_evaluations(b, r)
    m = c["metrics"][name]
    assert m["paired_count"] == 1 and m["baseline_paired_mean"] == 0.0
    assert m["reranker_paired_mean"] == m["mean_delta"] == 0.2
    assert m["completion"]["baseline"]["failed"] == 1
    assert m["completion"]["reranker"]["skipped"] == 1
    assert m["excluded_sample_ids"] == ["s2", "s3"]
    assert c["samples"][1]["errors"]["baseline"]
    assert c["samples"][1]["metric_deltas"][name] is None


def test_win_tie_loss_positive_negative_and_tolerance(runs):
    b, r = runs
    for row, value in zip(r["samples"], [0.8, 0.50000001, 0.1]):
        row["metrics"][METRICS[0]] = value
    m = compare_evaluations(b, r)["metrics"][METRICS[0]]
    assert (m["wins"], m["ties"], m["losses"]) == (1, 1, 1)
    assert m["mean_delta"] < 0


def test_context_order_sets_and_reference_coverage(runs):
    b, r = runs
    r["samples"][1]["retrieved_context_ids"] = ["b", "a"]
    r["samples"][1]["retrieved_contexts"] = ["b text", "a text"]
    r["samples"][2]["retrieved_context_ids"] = ["b", "c"]
    r["samples"][2]["retrieved_contexts"] = ["b text", "c text"]
    c = compare_evaluations(b, r)
    summary = c["retrieval_summary"]
    assert (summary["exact_same"], summary["same_set_different_order"], summary["different_set"]) == (1, 1, 1)
    assert summary["reference_coverage"] == dict(improved=0, unchanged=2, degraded=1, unavailable=0)
    assert c["samples"][2]["added_context_ids"] == ["c"]
    assert c["samples"][2]["removed_context_ids"] == ["a"]
    b["samples"][2], r["samples"][2] = r["samples"][2], b["samples"][2]
    assert compare_evaluations(b, r)["retrieval_summary"]["reference_coverage"]["improved"] == 1


def test_inputs_unchanged_and_report_contains_metrics_counts_limitations(runs):
    before = deepcopy(runs)
    c = compare_evaluations(*runs)
    report = render_report(c)
    assert runs == before
    assert all(name in report for name in METRICS)
    assert "Paired N" in report and "5 chunks" in report and "single-run" in report
    assert "None observed" in report


def test_no_success_pair_has_null_means(runs):
    for row in runs[0]["samples"]:
        row["metric_status"][METRICS[0]] = "failed"
        row["metrics"][METRICS[0]] = None
    m = compare_evaluations(*runs)["metrics"][METRICS[0]]
    assert m["paired_count"] == 0 and m["mean_delta"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0.5"])
def test_invalid_success_value_rejected(runs, value):
    runs[0]["samples"][0]["metrics"][METRICS[0]] = value
    with pytest.raises(ComparisonError, match="finite numeric"):
        compare_evaluations(*runs)


def test_ground_truth_mismatch_rejected(runs):
    runs[1]["samples"][0]["reference_answer"] = "changed"
    with pytest.raises(ComparisonError, match="Ground Truth"):
        compare_evaluations(*runs)


@pytest.mark.parametrize("damage", ["duplicate", "order", "count", "text", "final_outside_pool"])
def test_malformed_identity_and_order_rejected(runs, damage):
    b, r = runs
    if damage == "duplicate":
        r["samples"][1]["sample_id"] = "s1"
    elif damage == "order":
        r["samples"].reverse()
        r["metadata"]["sample_ids"].reverse()
    elif damage == "count":
        r["samples"][0]["actual_candidate_count"] = 99
    elif damage == "text":
        r["samples"][0]["retrieved_contexts"][0] = "different text"
    else:
        r["samples"][0]["retrieved_context_ids"][0] = "outside"
    with pytest.raises(ComparisonError):
        compare_evaluations(b, r)


def test_failed_retrieval_disclosed_as_unavailable(runs):
    for row in (runs[0]["samples"][0], runs[1]["samples"][0]):
        row.update(actual_candidate_count=None, candidate_context_ids=[],
                   retrieved_context_ids=[], retrieved_contexts=[], status="failed")
        row["metric_status"] = dict.fromkeys(METRICS, "skipped")
        row["metrics"] = dict.fromkeys(METRICS)
    c = compare_evaluations(*runs)
    assert c["retrieval_summary"]["reference_coverage"]["unavailable"] == 1
    assert not c["samples"][0]["reranker_attribution_valid"]
    assert c["metadata_validation"]["candidate_pool_issues"][0]["issue"] == "candidate_pool_unavailable"


def test_cli_and_comparison_load_no_models(runs, tmp_path):
    b, r, out, report = [tmp_path / name for name in ("b.json", "r.json", "c.json", "report.md")]
    for path, data in zip((b, r), runs):
        path.write_text(json.dumps(data), encoding="utf-8")
    script = """
import sys, runpy
class BlockModels:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'sentence_transformers','torch','chromadb','openai','ragas'}:
            raise AssertionError('Unexpected model import: ' + fullname)
sys.meta_path.insert(0, BlockModels())
runpy.run_module('scripts.compare_rag_evaluations', run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-B", "-c", script, "--baseline", str(b), "--reranker", str(r), "--output", str(out), "--report", str(report)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text())["metrics"][METRICS[0]]["paired_count"] == 3
    assert report.exists()
