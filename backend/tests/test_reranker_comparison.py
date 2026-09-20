from copy import deepcopy
from dataclasses import replace
import json

import pytest

from app.services.retrieval_service import RetrievalCandidate
from scripts.compare_reranker import (
    compare_rankings, find_rank, format_report, load_cases, summarize,
)


def candidates(ids):
    # Duplicate text and reversed distances ensure only ID and input order matter.
    return [RetrievalCandidate(record_id, "same text", {}, float(len(ids) - i))
            for i, record_id in enumerate(ids)]


@pytest.mark.parametrize("target,expected", [("A", 1), ("B", 2), ("C", 3), ("missing", None)])
def test_find_rank_uses_one_based_id_identity(target, expected):
    assert find_rank(candidates("ABC"), target) == expected


@pytest.mark.parametrize("before,after,target,delta,outcome", [
    ("ABCDE", "AEBCD", "E", 3, "IMPROVED"),
    ("ABCDE", "CBADE", "B", 0, "UNCHANGED"),
    ("ABCDE", "BCADE", "A", -2, "DEGRADED"),
])
def test_compare_rankings_classifies_all_outcomes(before, after, target, delta, outcome):
    baseline, reranked = candidates(before), candidates(after)
    original = deepcopy((baseline, reranked))
    result = compare_rankings(baseline, reranked, target, 3)
    assert result["rank_delta"] == delta
    assert result["outcome"] == outcome
    assert (baseline, reranked) == original


@pytest.mark.parametrize("target,hit", [("B", True), ("C", True), ("D", False)])
def test_top_n_hit_includes_boundary(target, hit):
    result = compare_rankings(candidates("ABCD"), candidates("ABCD"), target, 3)
    assert result["baseline_top_n_hit"] is hit
    assert result["reranked_top_n_hit"] is hit


def test_reranking_can_move_target_into_context():
    result = compare_rankings(candidates("ABCDE"), candidates("AEBCD"), "E", 3)
    assert result["baseline_rank"] == 5
    assert result["reranked_rank"] == 2
    assert not result["baseline_top_n_hit"]
    assert result["reranked_top_n_hit"]


@pytest.mark.parametrize("pool", ["ABC", ""])
def test_missing_target_is_not_retrieved_not_degraded(pool):
    result = compare_rankings(candidates(pool), candidates(pool[::-1]), "X", 3)
    assert result == {
        "baseline_rank": None, "reranked_rank": None, "rank_delta": None,
        "baseline_top_n_hit": False, "reranked_top_n_hit": False,
        "outcome": "NOT_RETRIEVED",
    }


@pytest.mark.parametrize("before,after", [("ABC", "AB"), ("ABC", "ABD"),
                                         ("AAB", "AB"), ("AB", "ABB")])
def test_comparison_rejects_unfair_or_duplicate_pools(before, after):
    with pytest.raises(ValueError, match="candidate IDs|Candidate IDs"):
        compare_rankings(candidates(before), candidates(after), "A", 3)


@pytest.mark.parametrize("top_n", [0, -1])
def test_comparison_rejects_nonpositive_n(top_n):
    with pytest.raises(ValueError, match="top_n"):
        compare_rankings([], [], "A", top_n)


def test_comparison_preserves_chroma_order_and_ignores_score_for_identity():
    baseline = candidates("ABC")
    reranked = [replace(baseline[1], reranker_score=10.0),
                replace(baseline[0], reranker_score=5.0),
                replace(baseline[2], reranker_score=-1.0)]
    result = compare_rankings(baseline, reranked, "B", 1)
    assert (result["baseline_rank"], result["reranked_rank"]) == (2, 1)
    assert [candidate.id for candidate in baseline] == list("ABC")


def test_summary_retains_all_outcomes_and_hits():
    results = [compare_rankings(candidates("ABCD"), candidates(order), target, 3)
               for order, target in [("DABC", "D"), ("ABCD", "B"),
                                     ("BCDA", "A"), ("ABCD", "X")]]
    assert summarize(results) == {
        "cases": 4, "IMPROVED": 1, "UNCHANGED": 1, "DEGRADED": 1, "NOT_RETRIEVED": 1,
        "baseline_top_n_hits": 2, "reranked_top_n_hits": 2,
    }


def test_fixed_corpus_is_small_unique_and_has_expected_ids():
    cases = load_cases()
    assert 8 <= len(cases["documents"]) <= 15
    assert 5 <= len(cases["queries"]) <= 8
    assert len({document["topic"] for document in cases["documents"]}) >= 3
    assert all(query["question"] and query["expected_relevant_id"] for query in cases["queries"])


@pytest.mark.parametrize("damage", ["duplicate_document", "duplicate_query", "unknown_expected"])
def test_load_cases_rejects_invalid_fixture(tmp_path, damage):
    cases = load_cases()
    if damage == "duplicate_document":
        cases["documents"].append(cases["documents"][0])
    elif damage == "duplicate_query":
        cases["queries"].append(cases["queries"][0])
    else:
        cases["queries"][0]["expected_relevant_id"] = "missing"
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    with pytest.raises(ValueError):
        load_cases(path)


def test_format_report_shows_all_results_and_missing_boundary():
    from dataclasses import asdict

    baseline = candidates("ABCD")
    results = [{"id": "Q1", "question": "问题", "expected_relevant_id": "X",
                **compare_rankings(baseline, baseline, "X", 3),
                "baseline": [asdict(candidate) for candidate in baseline],
                "reranked": [asdict(candidate) for candidate in baseline]}]
    report = {"environment": {"rerank_top_n": 3}, "results": results,
              "summary": summarize(results)}
    rendered = format_report(report)
    assert "Q1: 问题" in rendered
    assert "Expected: X" in rendered
    assert "*3. C" in rendered and " 4. D" in rendered
    assert "NOT_RETRIEVED" in rendered and "Candidate Retrieval recall problem" in rendered
    assert "delta=N/A" in rendered
    assert "Baseline" in rendered and "Reranked" in rendered
