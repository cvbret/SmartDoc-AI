"""Pure standard-library paired analysis of saved evaluation results."""

from copy import deepcopy
import math
from statistics import mean

METRICS = ("faithfulness", "answer_relevancy", "context_precision_llm",
           "context_recall_llm", "context_precision_id", "context_recall_id")
COMPATIBLE_FIELDS = (
    "dataset_version", "corpus_version", "corpus_sha256", "configured_sample_count",
    "retrieval_k", "configured_retrieval_k", "rerank_top_n", "answer_model",
    "judge_model", "evaluation_embedding_model", "ragas_version", "chunking",
    "retrieval_embedding_model", "retrieval_device", "evaluation_embedding_device",
    "evaluation_embedding_normalize", "judge_temperature", "judge_max_tokens",
    "judge_max_retries", "corpus_context_count", "dataset_sample_count",
    "python_version", "dependency_versions",
)
LIMITATIONS = (
    "Only 20 samples and 5 chunks; this is a narrow fixed-corpus evaluation.",
    "K=10 exceeds corpus size, so actual candidate count is 5; final Top-N is 3.",
    "Answer and Judge both use deepseek-chat and may share biases and blind spots.",
    "LLM metrics have Judge variability: this is a single-run observational comparison.",
    "One formal run per mode does not establish statistical significance or generalization.",
    "ID precision is reference-ID overlap, not answer accuracy; one reference in Top-3 yields 1/3.",
)


class ComparisonError(ValueError):
    """Saved runs cannot be compared under the required experimental conditions."""


def _samples(report):
    rows = report["samples"]
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ComparisonError("Duplicate sample IDs")
    if len(ids) != report["metadata"]["configured_sample_count"]:
        raise ComparisonError("Sample count differs from configured_sample_count")
    if ids != report["metadata"]["sample_ids"]:
        raise ComparisonError("Sample order differs from metadata sample_ids")
    for row in rows:
        if row["actual_candidate_count"] is not None and row["actual_candidate_count"] != len(row["candidate_context_ids"]):
            raise ComparisonError("Candidate count differs from candidate IDs")
        for field in ("candidate_context_ids", "retrieved_context_ids", "reference_context_ids"):
            if len(row[field]) != len(set(row[field])):
                raise ComparisonError(f"Duplicate IDs in {field}")
        if len(row["retrieved_context_ids"]) != len(row["retrieved_contexts"]):
            raise ComparisonError("Final context ID/text length mismatch")
        if not set(row["retrieved_context_ids"]) <= set(row["candidate_context_ids"]):
            raise ComparisonError("Final contexts outside candidate pool")
        for metric in METRICS:
            status, value = row["metric_status"][metric], row["metrics"][metric]
            if status not in ("success", "failed", "skipped"):
                raise ComparisonError("Unknown metric status")
            if status == "success":
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ComparisonError("Successful metric must be finite numeric")
            elif value is not None:
                raise ComparisonError("Unsuccessful metric must be null")
    return {row["sample_id"]: row for row in rows}


def _coverage(row):
    actual, reference = set(row["retrieved_context_ids"]), set(row["reference_context_ids"])
    return {"reference_hits": sorted(actual & reference),
            "missing_reference_ids": sorted(reference - actual),
            "extra_non_reference_ids": sorted(actual - reference)}


def compare_evaluations(baseline: dict, reranker: dict, tolerance: float = 1e-6) -> dict:
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ComparisonError("Tolerance must be finite and positive")
    bm, rm = baseline["metadata"], reranker["metadata"]
    if bm.get("run_mode") != "baseline" or rm.get("run_mode") != "reranker":
        raise ComparisonError("Expected baseline then reranker run modes")
    for field in COMPATIBLE_FIELDS:
        if field not in bm or field not in rm or bm[field] != rm[field]:
            raise ComparisonError(f"Incompatible metadata field: {field}")
    bs, rs = _samples(baseline), _samples(reranker)
    if set(bs) != set(rs):
        raise ComparisonError("Sample ID sets differ")
    if list(bs) != list(rs):
        raise ComparisonError("Sample order differs")
    rows, issues = [], []
    for sid, b in bs.items():
        r = rs[sid]
        for field in ("question", "reference_answer", "reference_context_ids", "reference_contexts"):
            if b[field] != r[field]:
                raise ComparisonError(f"Ground Truth mismatch: {sid} / {field}")
        # Shared IDs must also refer to identical text across runs.
        texts = {}
        for row in (b, r):
            for cid, text in zip(row["retrieved_context_ids"], row["retrieved_contexts"]):
                if cid in texts and texts[cid] != text:
                    raise ComparisonError(f"Context text mismatch: {sid}")
                texts[cid] = text
        pool_same = set(b["candidate_context_ids"]) == set(r["candidate_context_ids"])
        pool_observed = b["actual_candidate_count"] is not None and r["actual_candidate_count"] is not None
        if not pool_same or not pool_observed:
            issues.append({"sample_id": sid, "issue": "candidate_pool_mismatch" if not pool_same else "candidate_pool_unavailable"})
        b_ids, r_ids = b["retrieved_context_ids"], r["retrieved_context_ids"]
        bc, rc = _coverage(b), _coverage(r)
        coverage_delta = len(rc["reference_hits"]) - len(bc["reference_hits"])
        available = pool_observed and b["status"] != "failed" and r["status"] != "failed"
        rows.append({
            "sample_id": sid, "question": b["question"],
            "reference_context_ids": b["reference_context_ids"].copy(),
            "candidate_pool_same_set": pool_same, "candidate_pool_observed": pool_observed,
            "candidate_pool_same_order": b["candidate_context_ids"] == r["candidate_context_ids"],
            "reranker_attribution_valid": pool_same and available,
            "baseline_final_context_ids": b_ids.copy(), "reranker_final_context_ids": r_ids.copy(),
            "same_order": b_ids == r_ids, "same_set": set(b_ids) == set(r_ids),
            "added_context_ids": sorted(set(r_ids) - set(b_ids)),
            "removed_context_ids": sorted(set(b_ids) - set(r_ids)),
            "baseline_coverage": bc, "reranker_coverage": rc,
            "coverage_outcome": ("improved" if coverage_delta > 0 else "degraded" if coverage_delta < 0 else "unchanged") if available else "unavailable",
            "metric_deltas": {}, "errors": {"baseline": deepcopy(b["errors"]), "reranker": deepcopy(r["errors"])},
        })
    metrics = {}
    for metric in METRICS:
        pairs = [sid for sid in bs if bs[sid]["metric_status"][metric] == rs[sid]["metric_status"][metric] == "success"]
        bvalues = [bs[sid]["metrics"][metric] for sid in pairs]
        rvalues = [rs[sid]["metrics"][metric] for sid in pairs]
        deltas = [r - b for b, r in zip(bvalues, rvalues)]
        counts = {mode: {status: sum(row["metric_status"][metric] == status for row in samples.values())
                        for status in ("success", "failed", "skipped")}
                  for mode, samples in (("baseline", bs), ("reranker", rs))}
        metrics[metric] = {
            "configured_count": len(bs), "completion": counts, "paired_count": len(pairs),
            "paired_sample_ids": pairs, "excluded_sample_ids": [sid for sid in bs if sid not in pairs],
            "baseline_paired_mean": mean(bvalues) if pairs else None,
            "reranker_paired_mean": mean(rvalues) if pairs else None,
            "mean_delta": mean(rvalues) - mean(bvalues) if pairs else None,
            "wins": sum(delta >= tolerance for delta in deltas),
            "ties": sum(abs(delta) < tolerance for delta in deltas),
            "losses": sum(delta <= -tolerance for delta in deltas),
        }
        for row in rows:
            sid = row["sample_id"]
            row["metric_deltas"][metric] = rs[sid]["metrics"][metric] - bs[sid]["metrics"][metric] if sid in pairs else None
    return {
        "metadata_validation": {"compatible": True, "validated_fields": list(COMPATIBLE_FIELDS),
                                "sample_order_equal": True, "candidate_pool_issues": issues,
                                "reranker_attribution_valid": not issues},
        "setup": {field: deepcopy(bm[field]) for field in COMPATIBLE_FIELDS},
        "run_timestamps": {"baseline": bm["timestamp_utc"], "reranker": rm["timestamp_utc"]},
        "tolerance": tolerance, "limitations": list(LIMITATIONS),
        "retrieval_summary": {
            "sample_count": len(rows), "exact_same": sum(row["same_order"] for row in rows),
            "changed": sum(not row["same_order"] for row in rows),
            "same_set_different_order": sum(row["same_set"] and not row["same_order"] for row in rows),
            "different_set": sum(not row["same_set"] for row in rows),
            "candidate_same_set": sum(row["candidate_pool_same_set"] for row in rows),
            "candidate_same_order": sum(row["candidate_pool_same_order"] for row in rows),
            "reference_coverage": {outcome: sum(row["coverage_outcome"] == outcome for row in rows)
                                   for outcome in ("improved", "unchanged", "degraded", "unavailable")},
        },
        "metrics": metrics, "samples": rows,
        "original_summaries": {"baseline": deepcopy(baseline["summary"]), "reranker": deepcopy(reranker["summary"])},
    }


def metric_table(comparison):
    lines = ["| Metric | Baseline | Reranker | Delta | Paired N | Win | Tie | Loss |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, metric in comparison["metrics"].items():
        values = ["NULL" if metric[key] is None else f"{metric[key]:.6f}" for key in
                  ("baseline_paired_mean", "reranker_paired_mean", "mean_delta")]
        lines.append(f"| {name} | {' | '.join(values)} | {metric['paired_count']} | {metric['wins']} | {metric['ties']} | {metric['losses']} |")
    return "\n".join(lines)


def render_report(comparison):
    c, setup = comparison, comparison["setup"]
    lines = ["# SmartDoc-AI RAG Evaluation", "", "## Evaluation Setup", "",
             f"Dataset {setup['dataset_version']}; corpus {setup['corpus_version']}; {setup['configured_sample_count']} samples; {setup['corpus_context_count']} chunks.",
             f"Corpus SHA-256: `{setup['corpus_sha256']}`.",
             f"K={setup['retrieval_k']}, Top-N={setup['rerank_top_n']}; RAGAS {setup['ragas_version']}.",
             f"Retrieval/evaluation embedding: `{setup['retrieval_embedding_model']}` / `{setup['evaluation_embedding_model']}`; CPU normalized evaluation embeddings.",
             f"Answer/Judge: `{setup['answer_model']}` / `{setup['judge_model']}`. Judge temperature 0; Instructor JSON adapter with AsyncOpenAI.",
             "", "## Dataset", "", "Fixed TASK-004 Ground Truth; reference answers/IDs remain separate from runtime answers/contexts. Dataset order and shared production prompt were unchanged for both official runs.",
             "", "## Compared Pipelines", "", "Baseline: BGE-small → Ephemeral Chroma → original Top-3 → DeepSeek.",
             "Reranker: identical candidate retrieval → BAAI/bge-reranker-base → Top-3 → DeepSeek.",
             "", "## Metrics", "", "Four RAGAS collections semantic metrics and two deterministic ID overlap metrics. Only same-sample, same-metric successful pairs enter means; delta = reranker − baseline. Absolute delta below 1e-6 is a tie. ID precision is not LLM rank-sensitive semantic precision.",
             "", "## Aggregate Results", "", metric_table(c), "", "### Completion and Errors", "",
             "| Metric | Baseline success/failed/skipped | Reranker success/failed/skipped | Paired |", "|---|---:|---:|---:|"]
    for name, metric in c["metrics"].items():
        counts = ["/".join(str(metric["completion"][mode][status]) for status in ("success", "failed", "skipped")) for mode in ("baseline", "reranker")]
        lines.append(f"| {name} | {counts[0]} | {counts[1]} | {metric['paired_count']} |")
    errors = [(row["sample_id"], mode, error) for row in c["samples"] for mode in ("baseline", "reranker") for error in row["errors"][mode]]
    retrieval = c["retrieval_summary"]
    lines.extend(["", *( [f"- {sid}, {mode}: {error['stage']} / {error['error_type']} — {error['message']}" for sid, mode, error in errors] or ["Errors: None."] ),
                  "", "## Retrieval-Level Changes", "",
                  f"Final contexts: exact same {retrieval['exact_same']}; changed {retrieval['changed']}; same set/different order {retrieval['same_set_different_order']}; different set {retrieval['different_set']}.",
                  f"Candidate pool: same set {retrieval['candidate_same_set']}/{retrieval['sample_count']}; same order {retrieval['candidate_same_order']}/{retrieval['sample_count']}.",
                  "Reference coverage: " + "; ".join(f"{key} {value}" for key, value in retrieval['reference_coverage'].items()) + ".", "",
                  f"Candidate pool validity issues: {c['metadata_validation']['candidate_pool_issues']}. Any affected sample must not be attributed solely to reranking.",
                  "", "## Representative Cases", ""])
    selected = []
    for outcome in ("improved", "unchanged", "degraded"):
        matches = [row for row in c["samples"] if row["coverage_outcome"] == outcome]
        if matches:
            selected.append((f"Reference coverage {outcome}", matches[0]))
        else:
            lines.append(f"Reference coverage {outcome}: None observed.\n")
    # Include negative answer evidence even when deterministic reference coverage is unchanged.
    losses = [row for row in c["samples"] if any(row["metric_deltas"][m] is not None and row["metric_deltas"][m] < -c["tolerance"] for m in METRICS[:4]) and row not in [item[1] for item in selected]]
    if losses and len(selected) < 4:
        selected.append(("Observed semantic metric regression", losses[0]))
    gains = [row for row in c["samples"] if row["metric_deltas"]["context_precision_llm"] is not None and row["metric_deltas"]["context_precision_llm"] >= c["tolerance"] and row not in [item[1] for item in selected]]
    if gains and len(selected) < 4:
        selected.append(("Observed semantic context precision gain", gains[0]))
    unchanged_with_variation = [row for row in c["samples"] if row["same_order"] and
                               row["metric_deltas"]["context_precision_llm"] is not None and
                               abs(row["metric_deltas"]["context_precision_llm"]) >= c["tolerance"] and
                               row not in [item[1] for item in selected]]
    if unchanged_with_variation and len(selected) < 4:
        selected.append(("Identical contexts with Judge variation", unchanged_with_variation[0]))
    if len(selected) < 2:
        remaining = [row for row in c["samples"] if row not in [item[1] for item in selected]]
        if remaining:
            selected.append(("Additional context comparison", remaining[0]))
    for label, row in selected:
        lines.extend([f"### {row['sample_id']} — {label}", "", row["question"], "",
                      f"Reference IDs: `{row['reference_context_ids']}`.",
                      f"Baseline final IDs: `{row['baseline_final_context_ids']}`.",
                      f"Reranker final IDs: `{row['reranker_final_context_ids']}`.",
                      f"Reference coverage: {row['coverage_outcome']}; same set={row['same_set']}, same order={row['same_order']}.",
                      f"Absolute metric deltas: `{row['metric_deltas']}`.",
                      "Context selection/order and answer/Judge variability can affect semantic scores separately; this case does not establish a causal answer-quality gain.", ""])
    lines.extend(["## Interpretation", "",
                  "The deterministic evidence is the final-ID/order change and reference coverage shown above. Semantic score deltas are observations from independently generated answers and Judge calls, not isolated causal effects of reranking.",
                  "A higher rank-sensitive context precision score can coexist with unchanged reference coverage or a lower answer score. Even identical final contexts can receive different Judge scores in separate calls. Negative deltas are retained and no run was repeated to improve them.", "",
                  "## Limitations", "", *[f"- {item}" for item in LIMITATIONS], "", "## Reproduction", "",
                  "From `backend`, after installing the pinned requirements and configuring DeepSeek credentials:", "", "```powershell",
                  "New-Item -ItemType Directory -Force evaluation/results",
                  "..\\venv\\Scripts\\python.exe -B -m scripts.run_rag_evaluation --mode baseline --output evaluation/results/task006_baseline.json",
                  "..\\venv\\Scripts\\python.exe -B -m scripts.run_rag_evaluation --mode reranker --output evaluation/results/task006_reranker.json",
                  "..\\venv\\Scripts\\python.exe -B -m scripts.compare_rag_evaluations --baseline evaluation/results/task006_baseline.json --reranker evaluation/results/task006_reranker.json --output evaluation/results/task006_comparison.json --report evaluation/results/TASK006_EVALUATION_REPORT.md", "```", "",
                  "Each official mode was executed once. Reproduction invokes paid APIs; use new output filenames for future runs and retain this evidence. Comparison alone reads saved JSON and loads no models. Execution timing is stored in the corresponding `*_execution.json` files.", ""])
    return "\n".join(lines)
