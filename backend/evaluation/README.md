# Fixed RAG evaluation inputs — 1.0

These are independent, developer-authored knowledge documents and ground truth,
not renamed TASK-003 candidates or captured model answers. The HTTP profile API
and PostgreSQL/Redis cache workflow are explicitly scoped example conventions.
All answers must be assessed against these documents, not unstated outside facts.

From `backend`, run:

```powershell
..\venv\Scripts\python.exe -B -m scripts.inspect_eval_dataset
..\venv\Scripts\python.exe -B -m scripts.inspect_eval_dataset --sample EVAL-010 EVAL-002 EVAL-019
```

The loader returns `(dataset, contexts_by_id)`. A future runner can map question
to user input, reference_answer to reference, and resolve reference_context_ids
to reference texts. Actual answers, retrieved contexts and scores belong to that
future run, never to this fixed dataset. No models or databases are needed here.

## Reproducibility and ingestion boundary

Five complete short Markdown guides are UTF-8 decoded with CRLF/CR normalized to
LF. Production Markdown parsing is just UTF-8 decoding, but its module imports
model and persistent-vector initialization. The loader avoids those imports and
reuses production `split_text` with chunk_size=500, overlap=100. It does not run
the upload, PDF/DOCX, embedding, storage or answer pipeline.

Every current guide is under 500 characters, yielding one full-document context.
Four cross-document questions therefore genuinely need two distinct contexts.
This first version does not exercise answer spans crossing chunk boundaries;
that is a dataset limitation, not a reason to replace real splitting with manual
chunks. Tests separately protect repeatability on a longer document.

Context IDs are `<filename stem>::chunk_<zero-based index, three digits>`.
Production random Chroma UUIDs are never ground truth. A future isolated runner
must retain these IDs or map runtime records back to them.

Dataset and corpus versions are both 1.0. `split_text-v1` records the current
character-window algorithm. The corpus SHA-256 covers sorted `(filename, LF text)`
pairs serialized as compact UTF-8 JSON with ensure_ascii=False. It ignores
checkout newline differences, but detects source renames or content changes.
Loader rejects a hash, version or chunk-parameter mismatch. If documents,
chunking, parsing or samples change, review every affected answer and reference,
bump the appropriate version, update the supported contract and regenerate the
hash deliberately. A hash proves identity, not semantic correctness.

## Ground truth review

All 20 question/answer/reference triples were inspected against the resolved
texts. Direct answers are paraphrases, not mandatory output strings. In particular:

- EVAL-001–003: None return, preservation vs mutation, key/reverse/stability.
- EVAL-004–006: scoped PATCH/PUT rules and effect-based idempotence.
- EVAL-007–009: restart, image vs volume, deletion and backup limitations.
- EVAL-010–012: EXPIRE, PERSIST vs DEL, TTL sentinel meanings.
- EVAL-013–016: primary keys, transactions, tie-breaking order, index costs.
- EVAL-017: Python supplies non-mutating sorting; HTTP supplies read semantics.
- EVAL-018: HTTP supplies partial-update behavior; Redis supplies cache timing
  and the stale-cache failure window.
- EVAL-019: Docker supplies persistent storage; PostgreSQL supplies transactional
  grouping and business-condition checks. Neither alone supports the full answer.
- EVAL-020: PostgreSQL supplies the external-command atomicity boundary; Redis
  supplies the specific application cache-invalidating workflow and failure case.

Automated tests protect structural validity and selected semantic anchors, not
general truth entailment. Re-review remains necessary after content edits. This
dataset version contains answerable normal QA, with no runtime results or RAG metrics
stored in its ground truth.

## Offline evaluation runner (TASK-005)

Install the verified pins in `backend/requirements.txt` into the project venv and
run `python -m pip check`. RAGAS 0.4.3 requires its LangChain dependencies even
though this runner uses no LangChain/LangGraph orchestration. Community 0.4.1 is
pinned because 0.4.2 removed a module imported by RAGAS. Instructor/datasets and
the fsspec, jiter, rich, websockets compatibility pins are deliberate.

From `backend`:

```powershell
# Target one single-context and one multi-context sample, without editing data.
..\venv\Scripts\python.exe -B -m scripts.run_rag_evaluation --mode reranker --sample EVAL-010 EVAL-019 --limit 2 --output "$env:TEMP\rag-smoke.json"

# Future full runs: no --limit means all 20; these commands make paid API calls.
..\venv\Scripts\python.exe -B -m scripts.run_rag_evaluation --mode baseline --output "$env:TEMP\rag-baseline.json"
..\venv\Scripts\python.exe -B -m scripts.run_rag_evaluation --mode reranker --output "$env:TEMP\rag-reranker.json"
```

Without `--output`, only progress and the run metadata/summary are printed. The
CLI never automatically writes runtime results beside the dataset. `--sample`
filters stable IDs in dataset order; `--limit` then takes the first N. Exit codes
are 0 for all samples/metrics successful, 1 for partial results (still saved),
and 2 for setup, argument or output errors. Output parents must already exist.

The runner calls the validated loader, embeds its texts with the production
embedding function, and creates a uniquely named Ephemeral Chroma collection.
Record IDs are context IDs, not UUIDs. It queries min(configured K, index count),
normalizes candidates, optionally reranks, then selects Top-N. Only these selected
texts/IDs enter the shared production prompt and the context metrics. History is
empty for each sample. The ephemeral collection is cleaned up even on failure;
the persistent vector service is never imported. Baseline never invokes reranking.

Answer generation uses production `chat_with_llm` and the extracted, unchanged
`build_rag_messages` prompt. Judge setup uses a separate AsyncOpenAI client,
RAGAS `llm_factory`, provider=openai, adapter=instructor, temperature=0,
max_tokens=2048, timeout=60 seconds and finite Instructor max_retries=2. The
OpenAI Judge client has max_retries=0 to avoid stacked client retries. A single
event loop processes samples and metrics sequentially. AnswerRelevancy uses
strictness=3 and an independent CPU HuggingFaceEmbeddings provider for
BAAI/bge-small-zh-v1.5 with normalize_embeddings=True.

| Result metric | Inputs | Meaning |
| --- | --- | --- |
| faithfulness | question, actual answer, final contexts | Whether answer statements are supported; not answer correctness |
| answer_relevancy | question, actual answer; Judge + embeddings | Whether the response addresses the question; not factual correctness |
| context_precision_llm | question, reference answer, final contexts | Semantic, rank-sensitive average precision |
| context_recall_llm | question, reference answer, final contexts | Coverage of reference-answer information |
| context_precision_id | final IDs, reference IDs | Unique ID intersection / retrieved ID count |
| context_recall_id | final IDs, reference IDs | Unique ID intersection / reference ID count |

LLM Context Precision != ID Context Precision. ID metrics ignore ranking within
the same set and do not recognize unannotated semantic equivalents. Their legacy
SingleTurnSample/single_turn_ascore calls are isolated in one helper because
RAGAS 0.4.3 does not export them from collections. Modern metrics use per-sample
ascore, never ragas.evaluate with collections objects.

Each result separates references, actual retrieval/answer and metric judgments.
Metadata binds versions, corpus hash, chunking, mode, model roles and parameters.
candidate_context_ids describe Top-K diagnostics; retrieved_context_ids and texts
are the actual final context. `completed_sample_count` counts samples with all
six metrics successful; `processed_sample_count` counts every recorded sample,
including partial/failed samples. `configured_sample_count` is after filters.

Metric exceptions, nonnumeric values, NaN and infinities become null plus an error,
never zero. A genuine 0.0 remains successful. A failed metric does not stop other
metrics or samples. Retrieval/reranker failure skips all dependent metrics;
answer failure skips Faithfulness/AnswerRelevancy while the four context metrics
can still run. Empty retrieval preserves production empty-context answer behavior;
metrics may then fail according to their own contracts. Every metric summary
reports configured/successful/failed/skipped counts and a mean over successes
only (null if none). JSON serialization forbids NaN.

SDK exception strings can include credentials or HTTP bodies. Errors retain stage,
exception type and HTTP status when available, but deliberately omit raw exception
text. During execution raw library logging is disabled and restored on exit;
diagnostic errors live in the structured report. Run this CLI in a separate
process, not an API request. No API key is written to metadata. Dataset/index/model
setup errors produce a sanitized CLI error instead of fabricated sample results.

Answer and Judge currently both use deepseek-chat. Same-model judging may
introduce shared-model bias and common blind spots. The five-chunk corpus means
configured K=10 yields only five candidates. ID recall at candidate level can
saturate; final Top-3 evidence is more meaningful, but this remains a small corpus.
TASK-005 verifies execution; it makes no baseline-versus-reranker improvement
claim and does not execute TASK-006's full comparison.
# TASK-006 正式比较与证据

正式结果位于 [results](results/)：`task006_baseline.json`、`task006_reranker.json`、
`task006_comparison.json`、[TASK006_EVALUATION_REPORT.md](results/TASK006_EVALUATION_REPORT.md)。
两个 `*_execution.json` 保存运行起止 UTC、耗时及退出码。每种模式正式执行一次；原始结果保留各自 summary。

在 backend 目录执行全量评估时不传 `--limit` 或 `--sample`：

```powershell
New-Item -ItemType Directory -Force evaluation/results
..\venv\Scripts\python.exe -B -m scripts.run_rag_evaluation --mode baseline --output evaluation/results/task006_baseline.json
..\venv\Scripts\python.exe -B -m scripts.run_rag_evaluation --mode reranker --output evaluation/results/task006_reranker.json
```

上述命令会调用付费 API；未来复现请使用新的文件名，保留此次正式证据。仅重新分析已保存结果：

```powershell
..\venv\Scripts\python.exe -B -m scripts.compare_rag_evaluations --baseline evaluation/results/task006_baseline.json --reranker evaluation/results/task006_reranker.json --output evaluation/results/task006_comparison.json --report evaluation/results/TASK006_EVALUATION_REPORT.md
```

Comparison 只使用标准库，不加载模型、不建立 Chroma、不调用 API。它拒绝主要 metadata、样本集合/顺序或 Ground Truth 不一致的输入，记录候选池 set/order 公平性问题，以及最终 context 的增删和参考覆盖变化。

每项指标只比较两侧均成功的同一样本，披露 success/failed/skipped 与 paired N；空 paired 集合输出 null 均值，真实 0 保留。Delta 是 reranker − baseline 的绝对分差，`abs(delta) < 1e-6` 视为 tie。候选池不一致的样本保留但标记不可单独归因于 reranker。

这是 20 样本、5 chunks、K=10、Top-3 的单次观察比较。DeepSeek 同时回答与评判，LLM metrics 有波动；不能据此声称统计显著性或泛化提升。ID precision 表示参考 ID 占最终 context 的比例，不是回答准确率。完整数值、失败和正负案例见正式报告。
