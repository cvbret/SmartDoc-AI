# SmartDoc-AI RAG Evaluation

## Evaluation Setup

Dataset 1.0; corpus 1.0; 20 samples; 5 chunks.
Corpus SHA-256: `98c73e903a24385a1078742c0cd42006e445693600347228b7c7efec261f645f`.
K=10, Top-N=3; RAGAS 0.4.3.
Retrieval/evaluation embedding: `BAAI/bge-small-zh-v1.5` / `BAAI/bge-small-zh-v1.5`; CPU normalized evaluation embeddings.
Answer/Judge: `deepseek-chat` / `deepseek-chat`. Judge temperature 0; Instructor JSON adapter with AsyncOpenAI.

## Dataset

Fixed TASK-004 Ground Truth; reference answers/IDs remain separate from runtime answers/contexts. Dataset order and shared production prompt were unchanged for both official runs.

## Compared Pipelines

Baseline: BGE-small → Ephemeral Chroma → original Top-3 → DeepSeek.
Reranker: identical candidate retrieval → BAAI/bge-reranker-base → Top-3 → DeepSeek.

## Metrics

Four RAGAS collections semantic metrics and two deterministic ID overlap metrics. Only same-sample, same-metric successful pairs enter means; delta = reranker − baseline. Absolute delta below 1e-6 is a tie. ID precision is not LLM rank-sensitive semantic precision.

## Aggregate Results

| Metric | Baseline | Reranker | Delta | Paired N | Win | Tie | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| faithfulness | 0.927579 | 0.894762 | -0.032817 | 20 | 2 | 13 | 5 |
| answer_relevancy | 0.843333 | 0.840149 | -0.003185 | 20 | 11 | 3 | 6 |
| context_precision_llm | 0.950000 | 1.000000 | 0.050000 | 20 | 3 | 17 | 0 |
| context_recall_llm | 1.000000 | 1.000000 | 0.000000 | 20 | 0 | 20 | 0 |
| context_precision_id | 0.400000 | 0.400000 | 0.000000 | 20 | 0 | 20 | 0 |
| context_recall_id | 1.000000 | 1.000000 | 0.000000 | 20 | 0 | 20 | 0 |

### Completion and Errors

| Metric | Baseline success/failed/skipped | Reranker success/failed/skipped | Paired |
|---|---:|---:|---:|
| faithfulness | 20/0/0 | 20/0/0 | 20 |
| answer_relevancy | 20/0/0 | 20/0/0 | 20 |
| context_precision_llm | 20/0/0 | 20/0/0 | 20 |
| context_recall_llm | 20/0/0 | 20/0/0 | 20 |
| context_precision_id | 20/0/0 | 20/0/0 | 20 |
| context_recall_id | 20/0/0 | 20/0/0 | 20 |

Errors: None.

## Retrieval-Level Changes

Final contexts: exact same 4; changed 16; same set/different order 5; different set 11.
Candidate pool: same set 20/20; same order 20/20.
Reference coverage: improved 0; unchanged 20; degraded 0; unavailable 0.

Candidate pool validity issues: []. Any affected sample must not be attributed solely to reranking.

## Representative Cases

Reference coverage improved: None observed.

Reference coverage degraded: None observed.

### EVAL-001 — Reference coverage unchanged

Python 原地排序列表的方法返回什么？

Reference IDs: `['python::chunk_000']`.
Baseline final IDs: `['python::chunk_000', 'postgresql::chunk_000', 'http::chunk_000']`.
Reranker final IDs: `['python::chunk_000', 'redis::chunk_000', 'postgresql::chunk_000']`.
Reference coverage: unchanged; same set=False, same order=False.
Absolute metric deltas: `{'faithfulness': 0.0, 'answer_relevancy': 0.0, 'context_precision_llm': 0.0, 'context_recall_llm': 0.0, 'context_precision_id': 0.0, 'context_recall_id': 0.0}`.
Context selection/order and answer/Judge variability can affect semantic scores separately; this case does not establish a causal answer-quality gain.

### EVAL-003 — Observed semantic metric regression

字符串列表按长度降序排列时如何设置排序参数？等长字符串的相对顺序会怎样？

Reference IDs: `['python::chunk_000']`.
Baseline final IDs: `['python::chunk_000', 'postgresql::chunk_000', 'http::chunk_000']`.
Reranker final IDs: `['python::chunk_000', 'postgresql::chunk_000', 'redis::chunk_000']`.
Reference coverage: unchanged; same set=False, same order=False.
Absolute metric deltas: `{'faithfulness': -0.5714285714285714, 'answer_relevancy': 0.0772690212847662, 'context_precision_llm': 0.0, 'context_recall_llm': 0.0, 'context_precision_id': 0.0, 'context_recall_id': 0.0}`.
Context selection/order and answer/Judge variability can affect semantic scores separately; this case does not establish a causal answer-quality gain.

### EVAL-012 — Observed semantic context precision gain

TTL 查询分别返回 -1 和 -2 时代表什么？

Reference IDs: `['redis::chunk_000']`.
Baseline final IDs: `['postgresql::chunk_000', 'http::chunk_000', 'redis::chunk_000']`.
Reranker final IDs: `['redis::chunk_000', 'postgresql::chunk_000', 'http::chunk_000']`.
Reference coverage: unchanged; same set=True, same order=False.
Absolute metric deltas: `{'faithfulness': 0.0, 'answer_relevancy': -0.30740015183425495, 'context_precision_llm': 0.6666666666000001, 'context_recall_llm': 0.0, 'context_precision_id': 0.0, 'context_recall_id': 0.0}`.
Context selection/order and answer/Judge variability can affect semantic scores separately; this case does not establish a causal answer-quality gain.

### EVAL-018 — Identical contexts with Judge variation

按示例应用约定，只改昵称并保留头像后，资料提交成功应该怎样处理 Redis 缓存？删除缓存失败有什么影响？

Reference IDs: `['http::chunk_000', 'redis::chunk_000']`.
Baseline final IDs: `['redis::chunk_000', 'http::chunk_000', 'postgresql::chunk_000']`.
Reranker final IDs: `['redis::chunk_000', 'http::chunk_000', 'postgresql::chunk_000']`.
Reference coverage: unchanged; same set=True, same order=True.
Absolute metric deltas: `{'faithfulness': 0.0, 'answer_relevancy': -0.05940333809454046, 'context_precision_llm': 0.1666666666083334, 'context_recall_llm': 0.0, 'context_precision_id': 0.0, 'context_recall_id': 0.0}`.
Context selection/order and answer/Judge variability can affect semantic scores separately; this case does not establish a causal answer-quality gain.

## Interpretation

The deterministic evidence is the final-ID/order change and reference coverage shown above. Semantic score deltas are observations from independently generated answers and Judge calls, not isolated causal effects of reranking.
A higher rank-sensitive context precision score can coexist with unchanged reference coverage or a lower answer score. Even identical final contexts can receive different Judge scores in separate calls. Negative deltas are retained and no run was repeated to improve them.

## Limitations

- Only 20 samples and 5 chunks; this is a narrow fixed-corpus evaluation.
- K=10 exceeds corpus size, so actual candidate count is 5; final Top-N is 3.
- Answer and Judge both use deepseek-chat and may share biases and blind spots.
- LLM metrics have Judge variability: this is a single-run observational comparison.
- One formal run per mode does not establish statistical significance or generalization.
- ID precision is reference-ID overlap, not answer accuracy; one reference in Top-3 yields 1/3.

## Reproduction

From `backend`, after installing the pinned requirements and configuring DeepSeek credentials:

```powershell
New-Item -ItemType Directory -Force evaluation/results
..\venv\Scripts\python.exe -B -m scripts.run_rag_evaluation --mode baseline --output evaluation/results/task006_baseline.json
..\venv\Scripts\python.exe -B -m scripts.run_rag_evaluation --mode reranker --output evaluation/results/task006_reranker.json
..\venv\Scripts\python.exe -B -m scripts.compare_rag_evaluations --baseline evaluation/results/task006_baseline.json --reranker evaluation/results/task006_reranker.json --output evaluation/results/task006_comparison.json --report evaluation/results/TASK006_EVALUATION_REPORT.md
```

Each official mode was executed once. Reproduction invokes paid APIs; use new output filenames for future runs and retain this evidence. Comparison alone reads saved JSON and loads no models. Execution timing is stored in the corresponding `*_execution.json` files.
