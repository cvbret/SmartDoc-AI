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
version contains answerable normal QA, with no runtime results or RAG metrics.
