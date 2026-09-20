from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock
import sys
import types

import numpy as np
import pytest

from app.services import reranker_service as service
from app.services.retrieval_service import RetrievalCandidate


@pytest.fixture
def candidates():
    return [RetrievalCandidate(name, text, {"source": name}, distance)
            for name, text, distance in [("A", "same", 0.3), ("B", "same", 0.2),
                                         ("C", "other", 0.1)]]


def mock_scores(monkeypatch, scores):
    model = Mock()
    model.predict.return_value = scores
    loader = Mock(return_value=model)
    monkeypatch.setattr(service, "get_reranker_model", loader)
    return loader, model


def test_pairs_scores_identity_duplicates_and_immutability(monkeypatch, candidates):
    before = deepcopy(candidates)
    _, model = mock_scores(monkeypatch, np.array([-1.0, 3.0, 0.5]))
    result = service.rerank_candidates("query", candidates)
    model.predict.assert_called_once_with(
        [("query", "same"), ("query", "same"), ("query", "other")],
        convert_to_numpy=True, show_progress_bar=False, apply_softmax=False,
    )
    assert [item.id for item in result] == ["B", "C", "A"]
    for item, index in zip(result, [1, 2, 0]):
        original = before[index]
        assert (item.id, item.text, item.metadata, item.retrieval_distance) == (
            original.id, original.text, original.metadata, original.retrieval_distance)
        assert item.reranker_score == [-1.0, 3.0, 0.5][index]
        assert type(item.reranker_score) is float
        assert item is not candidates[index]
    assert candidates == before


def test_stable_tie_ignores_distance(monkeypatch, candidates):
    mock_scores(monkeypatch, [2.0, 2.0, 2.0])
    assert [c.id for c in service.rerank_candidates("q", candidates)] == ["A", "B", "C"]


def test_empty_skips_loader_and_predict(monkeypatch):
    loader, model = mock_scores(monkeypatch, [])
    assert service.rerank_candidates("q", []) == []
    loader.assert_not_called()
    model.predict.assert_not_called()


def test_single_candidate(monkeypatch, candidates):
    mock_scores(monkeypatch, np.array([-8.0], dtype=np.float32))
    result = service.rerank_candidates("q", candidates[:1])
    assert len(result) == 1
    assert result[0].reranker_score == -8.0


def test_inference_failure_is_explicit(monkeypatch, candidates):
    _, model = mock_scores(monkeypatch, [])
    failure = RuntimeError("inference error")
    model.predict.side_effect = failure
    with pytest.raises(service.RerankerError, match="inference") as exc:
        service.rerank_candidates("q", candidates)
    assert exc.value.__cause__ is failure


@pytest.mark.parametrize("scores", [[1, 2], [1, 2, 3, 4], 1.0, [[1], [2], [3]],
                                    [[1, 2], [3, 4], [5, 6]]])
def test_invalid_score_shape(monkeypatch, candidates, scores):
    mock_scores(monkeypatch, scores)
    with pytest.raises(service.RerankerError, match="shape"):
        service.rerank_candidates("q", candidates)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"),
                                 "bad", None, 1j, True])
def test_invalid_score_value(monkeypatch, candidates, bad):
    scores = [bad, bad, bad] if isinstance(bad, bool) else [1, bad, 2]
    mock_scores(monkeypatch, scores)
    with pytest.raises(service.RerankerError):
        service.rerank_candidates("q", candidates)


@pytest.fixture
def constructor(monkeypatch):
    # Isolate loader tests from the suite's import-time SentenceTransformer double.
    fake = types.ModuleType("sentence_transformers")
    fake.CrossEncoder = Mock()
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    fake_nn = types.ModuleType("torch.nn")
    fake_nn.Identity = Mock(return_value=object())
    monkeypatch.setitem(sys.modules, "torch.nn", fake_nn)
    service._load_reranker_model.cache_clear()
    yield fake.CrossEncoder, fake_nn.Identity.return_value
    service._load_reranker_model.cache_clear()


def test_lazy_loader_reuses_model_and_preserves_logits(constructor):
    create, identity = constructor
    create.assert_not_called()
    with ThreadPoolExecutor(max_workers=4) as pool:
        models = list(pool.map(lambda _: service.get_reranker_model(), range(8)))
    assert all(model is create.return_value for model in models)
    create.assert_called_once_with(service.settings.reranker_model,
                                   activation_fn=identity, max_length=512)


def test_load_failure_is_explicit_and_retryable(constructor):
    create, _ = constructor
    failure = OSError("model unavailable")
    create.side_effect = failure
    with pytest.raises(service.RerankerError, match="load reranker") as exc:
        service.get_reranker_model()
    assert exc.value.__cause__ is failure
    create.side_effect = None
    assert service.get_reranker_model() is create.return_value
    assert create.call_count == 2
