from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import pytest

from evaluation.dataset import (
    ROOT, dataset_statistics, load_corpus, load_evaluation_dataset, validate_dataset,
)


@pytest.fixture
def loaded():
    return load_evaluation_dataset()


def test_fixed_dataset_loads_with_balanced_coverage_and_real_contexts(loaded):
    data, contexts = loaded
    stats = dataset_statistics(data, contexts)
    assert 15 <= stats["sample_count"] <= 30
    assert stats["sample_count"] == 20
    assert stats["category_distribution"] == dict.fromkeys(
        ["python", "http", "docker", "redis", "postgresql"], 4,
    )
    assert stats["difficulty_distribution"] == {"easy": 6, "medium": 10, "hard": 4}
    assert stats["type_distribution"] == {
        "direct_fact": 6, "fine_grained": 5, "multi_sentence": 5, "multi_context": 4,
    }
    assert stats["single_context_count"] == 16
    assert stats["multi_context_count"] == 4
    assert stats["document_count"] == stats["chunk_count"] == 5
    for sample in data["samples"]:
        assert all(contexts[ref].strip() for ref in sample["reference_context_ids"])


@pytest.mark.parametrize("field", ["id", "question"])
def test_duplicate_sample_identity_or_question_rejected(loaded, field):
    data, contexts = loaded
    data["samples"][1][field] = data["samples"][0][field]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_dataset(data, contexts, data["corpus_sha256"])


@pytest.mark.parametrize("field", ["id", "question", "reference_answer"])
@pytest.mark.parametrize("value", ["", " \n", None, 7])
def test_empty_or_nonstring_required_fields_rejected(loaded, field, value):
    data, contexts = loaded
    data["samples"][0][field] = value
    with pytest.raises(ValueError, match="nonempty string"):
        validate_dataset(data, contexts, data["corpus_sha256"])


@pytest.mark.parametrize("refs,message", [
    ([], "nonempty list"), (None, "nonempty list"), ("python::chunk_000", "nonempty list"),
    (["unknown::chunk_000"], "unknown context"), ([None], "nonempty string"),
    (["python::chunk_000", "python::chunk_000"], "duplicate context"),
])
def test_invalid_reference_contexts_rejected(loaded, refs, message):
    data, contexts = loaded
    data["samples"][0]["reference_context_ids"] = refs
    with pytest.raises(ValueError, match=message):
        validate_dataset(data, contexts, data["corpus_sha256"])


@pytest.mark.parametrize("field", ["dataset_version", "corpus_version"])
@pytest.mark.parametrize("value", [None, "", "2.0"])
def test_missing_or_unsupported_versions_rejected(loaded, field, value):
    data, contexts = loaded
    if value is None:
        del data[field]
    else:
        data[field] = value
    with pytest.raises(ValueError, match=field):
        validate_dataset(data, contexts, data["corpus_sha256"])


@pytest.mark.parametrize("field", ["category", "difficulty", "type"])
def test_invalid_classification_rejected(loaded, field):
    data, contexts = loaded
    data["samples"][0][field] = "unknown"
    with pytest.raises(ValueError, match=field):
        validate_dataset(data, contexts, data["corpus_sha256"])


def test_type_must_match_reference_count(loaded):
    data, contexts = loaded
    data["samples"][-1]["reference_context_ids"] = ["redis::chunk_000"]
    with pytest.raises(ValueError, match="context count"):
        validate_dataset(data, contexts, data["corpus_sha256"])


@pytest.mark.parametrize("field,value", [("chunk_size", 400), ("overlap", 50), ("version", "v2")])
def test_changed_chunking_contract_rejected(loaded, field, value):
    data, contexts = loaded
    data["chunking"][field] = value
    with pytest.raises(ValueError, match="chunking"):
        validate_dataset(data, contexts, data["corpus_sha256"])


def test_changed_corpus_invalidates_bound_dataset(tmp_path):
    for source in (ROOT / "corpus").glob("*.md"):
        (tmp_path / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    with (tmp_path / "redis.md").open("a", encoding="utf-8") as output:
        output.write("\n新增事实。\n")
    with pytest.raises(ValueError, match="corpus_sha256 mismatch"):
        load_evaluation_dataset(corpus_dir=tmp_path)


def test_context_ids_and_hash_repeat_across_checkout_newlines(tmp_path):
    expected = load_corpus()
    for source in (ROOT / "corpus").glob("*.md"):
        text = source.read_text(encoding="utf-8")
        (tmp_path / source.name).write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    assert load_corpus(tmp_path) == expected
    assert load_corpus(tmp_path) == load_corpus(tmp_path)


def test_long_document_uses_real_500_100_character_windows(tmp_path):
    text = "甲" * 400 + "乙" * 400 + "丙" * 250
    (tmp_path / "long.md").write_text(text, encoding="utf-8")
    contexts, _ = load_corpus(tmp_path)
    assert contexts == {
        "long::chunk_000": text[:500],
        "long::chunk_001": text[400:900],
        "long::chunk_002": text[800:1300],
    }


@pytest.mark.parametrize("empty_file", [False, True])
def test_missing_or_empty_corpus_rejected(tmp_path, empty_file):
    if empty_file:
        (tmp_path / "empty.md").write_text(" \n", encoding="utf-8")
    with pytest.raises(ValueError, match="no Markdown|Empty corpus"):
        load_corpus(tmp_path)


@pytest.mark.parametrize("sample_id,expected,anchors", [
    ("EVAL-010", ["redis::chunk_000"], ["EXPIRE profile 60", "期限内", "到期后自动失效"]),
    ("EVAL-002", ["python::chunk_000"], ["sorted", "原列表", "反转"]),
    ("EVAL-006", ["http::chunk_000"], ["PATCH", "幂等", "加一", "状态码"]),
    ("EVAL-019", ["docker::chunk_000", "postgresql::chunk_000"], ["命名卷", "扣款", "事务", "业务条件"]),
    ("EVAL-020", ["redis::chunk_000", "postgresql::chunk_000"], ["Redis", "提交成功", "TTL", "旧缓存"]),
])
def test_reviewed_semantic_anchors_have_the_required_sources(loaded, sample_id, expected, anchors):
    data, contexts = loaded
    sample = next(sample for sample in data["samples"] if sample["id"] == sample_id)
    assert sample["reference_context_ids"] == expected
    evidence = "\n".join(contexts[ref] for ref in expected)
    for anchor in anchors:
        assert anchor in evidence
        assert anchor in sample["reference_answer"]


def test_validation_does_not_mutate_ground_truth(loaded):
    data, contexts = loaded
    original = deepcopy((data, contexts))
    validate_dataset(data, contexts, data["corpus_sha256"])
    assert (data, contexts) == original
    forbidden = {"response", "actual_answer", "retrieved_contexts", "reranker_score"}
    assert all(not forbidden.intersection(sample) for sample in data["samples"])


def test_inspection_runs_without_models_database_or_settings():
    backend = Path(__file__).resolve().parents[1]
    code = '''
import sys
from scripts.inspect_eval_dataset import main
sys.argv = ["inspect_eval_dataset", "--sample", "EVAL-010"]
main()
assert not {"chromadb", "sentence_transformers", "app.core.config", "app.services.document_service"}.intersection(sys.modules)
'''
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=backend,
                            env=env, text=True, encoding="utf-8", capture_output=True, check=True)
    assert '"sample_count": 20' in result.stdout
    assert "Reference Answer:" in result.stdout
    assert "redis::chunk_000" in result.stdout
    assert "EXPIRE profile 60" in result.stdout
    assert "EVAL-001 [" not in result.stdout
