"""Load fixed ground truth without importing models, databases or RAG services."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from app.services.chunk_service import split_text


ROOT = Path(__file__).resolve().parent
CATEGORIES = {"python", "http", "docker", "redis", "postgresql"}
DIFFICULTIES = {"easy", "medium", "hard"}
TYPES = {"direct_fact", "fine_grained", "multi_sentence", "multi_context"}
CHUNKING = {"version": "split_text-v1", "chunk_size": 500, "overlap": 100}


def load_corpus(corpus_dir: Path = ROOT / "corpus") -> tuple[dict[str, str], str]:
    """Canonical UTF-8/LF Markdown, then the production character splitter.

    document_service's Markdown parser only decodes UTF-8, but importing it
    initializes embedding and persistent Chroma. Keep this decoding local.
    Normalize checkout line endings so Windows and Linux yield identical IDs.
    """
    paths = sorted(corpus_dir.glob("*.md"))
    if not paths:
        raise ValueError("Evaluation corpus has no Markdown documents")
    documents = []
    contexts = {}
    for index, path in enumerate(paths):
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"Empty corpus document: {path.name}")
        documents.append((path.name, text))
        chunks = split_text(text, path.name, index, chunk_size=500, overlap=100)
        for chunk in chunks:
            context_id = f"{path.stem}::chunk_{chunk['metadata']['chunk_id']:03d}"
            contexts[context_id] = chunk["content"]
    serialized = json.dumps(documents, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return contexts, digest


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value.strip()


def validate_dataset(data: dict, contexts: dict[str, str], corpus_hash: str) -> None:
    if not isinstance(data, dict):
        raise ValueError("Dataset must be an object")
    for field in ("dataset_version", "corpus_version"):
        if data.get(field) != "1.0":
            raise ValueError(f"Unsupported or missing {field}; expected 1.0")
    if data.get("corpus_sha256") != corpus_hash:
        raise ValueError("corpus_sha256 mismatch; review corpus and ground truth together")
    if data.get("chunking") != CHUNKING:
        raise ValueError("Unsupported chunking parameters or version")
    samples = data.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("samples must be a nonempty list")
    ids, questions = set(), set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Each sample must be an object")
        sample_id = _nonempty_string(sample.get("id"), "sample id")
        if sample_id in ids:
            raise ValueError(f"Duplicate sample ID: {sample_id}")
        ids.add(sample_id)
        question = _nonempty_string(sample.get("question"), f"{sample_id} question")
        if question in questions:
            raise ValueError(f"Duplicate question: {sample_id}")
        questions.add(question)
        _nonempty_string(sample.get("reference_answer"), f"{sample_id} reference_answer")
        for field, allowed in (("category", CATEGORIES), ("difficulty", DIFFICULTIES), ("type", TYPES)):
            value = sample.get(field)
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(f"{sample_id}: invalid {field}")
        refs = sample.get("reference_context_ids")
        if not isinstance(refs, list) or not refs:
            raise ValueError(f"{sample_id}: reference_context_ids must be a nonempty list")
        for ref in refs:
            _nonempty_string(ref, f"{sample_id} context ID")
            if ref not in contexts:
                raise ValueError(f"{sample_id}: unknown context ID {ref}")
        if len(set(refs)) != len(refs):
            raise ValueError(f"{sample_id}: duplicate context IDs")
        if (sample["type"] == "multi_context") != (len(refs) > 1):
            raise ValueError(f"{sample_id}: type and context count disagree")


def load_evaluation_dataset(
    dataset_path: Path = ROOT / "rag_eval_dataset.json",
    corpus_dir: Path = ROOT / "corpus",
) -> tuple[dict, dict[str, str]]:
    """Return static dataset and an ID-to-text map for reference contexts."""
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    contexts, digest = load_corpus(corpus_dir)
    validate_dataset(data, contexts, digest)
    return data, contexts


def dataset_statistics(data: dict, contexts: dict[str, str]) -> dict:
    samples = data["samples"]
    multi = sum(len(sample["reference_context_ids"]) > 1 for sample in samples)
    return {
        "dataset_version": data["dataset_version"], "corpus_version": data["corpus_version"],
        "corpus_sha256": data["corpus_sha256"], "chunking": data["chunking"],
        "sample_count": len(samples),
        "category_distribution": dict(Counter(sample["category"] for sample in samples)),
        "difficulty_distribution": dict(Counter(sample["difficulty"] for sample in samples)),
        "type_distribution": dict(Counter(sample["type"] for sample in samples)),
        "single_context_count": len(samples) - multi, "multi_context_count": multi,
        "document_count": len({context_id.split("::")[0] for context_id in contexts}),
        "chunk_count": len(contexts),
    }
