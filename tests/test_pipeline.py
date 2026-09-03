"""
Tests for the parts of the pipeline that do not need model weights.

A stub embedder stands in for the real model so retrieval, prompt construction and
the abstention path can be checked deterministically and in under a second. The
pooling function is tested against hand-computed values, since that is the piece
that was silently wrong in the original inline implementation.

Usage:
    python -m pytest tests/ -v
    python tests/test_pipeline.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings import mean_pool  # noqa: E402
from vector_store import LocalVectorStore  # noqa: E402
from chatbot import build_prompt, format_context, SupportChatbot, NO_ANSWER  # noqa: E402


# --- pooling -------------------------------------------------------------------

def test_mean_pool_ignores_padding():
    """Two real tokens then two padding tokens should average only the real ones."""
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0, 0]])
    result = mean_pool(hidden, mask)
    assert torch.allclose(result, torch.tensor([[2.0, 2.0]])), result


def test_mean_pool_differs_from_naive_mean():
    """The naive torch.mean the original code used gives a different, wrong answer."""
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [0.0, 0.0], [0.0, 0.0]]])
    mask = torch.tensor([[1, 1, 0, 0]])
    correct = mean_pool(hidden, mask)
    naive = torch.mean(hidden, dim=1)
    assert torch.allclose(correct, torch.tensor([[2.0, 2.0]]))
    assert torch.allclose(naive, torch.tensor([[1.0, 1.0]]))
    assert not torch.allclose(correct, naive)


def test_mean_pool_all_real_tokens_matches_plain_mean():
    """With no padding, masked pooling and a plain mean must agree."""
    hidden = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.long)
    assert torch.allclose(mean_pool(hidden, mask), hidden.mean(dim=1), atol=1e-6)


# --- vector store --------------------------------------------------------------

def _toy_store() -> LocalVectorStore:
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.7071, 0.7071, 0.0],
        ],
        dtype=np.float32,
    )
    metadata = [
        {"ticket_id": "A", "subject": "x axis", "messages": "about x"},
        {"ticket_id": "B", "subject": "y axis", "messages": "about y"},
        {"ticket_id": "C", "subject": "z axis", "messages": "about z"},
        {"ticket_id": "D", "subject": "xy diagonal", "messages": "about xy"},
    ]
    store = LocalVectorStore()
    store.upsert(vectors, metadata)
    return store


def test_query_ranks_by_cosine_similarity():
    store = _toy_store()
    results = store.query(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=4)
    assert [r["ticket_id"] for r in results] == ["A", "D", "B", "C"] or [
        r["ticket_id"] for r in results
    ][:2] == ["A", "D"]
    assert results[0]["score"] > results[1]["score"] > results[2]["score"] - 1e-6


def test_query_normalizes_the_input_vector():
    """An unnormalized query must rank identically to its normalized form."""
    store = _toy_store()
    a = store.query(np.array([5.0, 0.0, 0.0], dtype=np.float32), top_k=4)
    b = store.query(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=4)
    assert [r["ticket_id"] for r in a] == [r["ticket_id"] for r in b]
    assert abs(a[0]["score"] - b[0]["score"]) < 1e-6


def test_query_respects_top_k_and_empty_store():
    store = _toy_store()
    assert len(store.query(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=2)) == 2
    assert LocalVectorStore().query(np.array([1.0, 0.0, 0.0], dtype=np.float32)) == []


def test_upsert_rejects_length_mismatch():
    store = LocalVectorStore()
    try:
        store.upsert(np.zeros((3, 4), dtype=np.float32), [{"ticket_id": "A"}])
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched lengths")


def test_save_and_load_roundtrip():
    store = _toy_store()
    with tempfile.TemporaryDirectory() as tmp:
        store.save(tmp)
        loaded = LocalVectorStore.load(tmp)
    assert len(loaded) == len(store)
    assert np.allclose(loaded.vectors, store.vectors)
    assert loaded.metadata == store.metadata


# --- prompt construction -------------------------------------------------------

def test_build_prompt_includes_question_and_tickets():
    results = [
        {"ticket_id": "1001", "subject": "Upload size", "messages": "raise the limit", "score": 0.8},
        {"ticket_id": "1002", "subject": "SSL", "messages": "cert renewed", "score": 0.5},
    ]
    messages = build_prompt("How do I upload bigger files?", results)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    user = messages[1]["content"]
    assert "How do I upload bigger files?" in user
    assert "1001" in user and "1002" in user
    assert "raise the limit" in user


def test_format_context_truncates_long_messages():
    long_ticket = [{"ticket_id": "1", "subject": "s", "messages": "x" * 5000, "score": 1.0}]
    context = format_context(long_ticket)
    assert "[truncated]" in context
    assert len(context) < 2000


# --- abstention ----------------------------------------------------------------

class StubEmbedder:
    """Returns a fixed vector so retrieval scores are controllable in tests."""

    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32)

    def encode_one(self, text, normalize=True):
        return self.vector

    def encode(self, texts, batch_size=16, normalize=True):
        return np.vstack([self.vector for _ in texts])


def test_abstains_when_best_score_is_below_threshold():
    """A query orthogonal to everything indexed must not reach the generator."""
    with tempfile.TemporaryDirectory() as tmp:
        _toy_store().save(tmp)
        bot = SupportChatbot(
            index_dir=tmp,
            embedder=StubEmbedder([0.0, 0.0, 0.0]),
            load_generator=False,
        )
        result = bot.answer("something unrelated")
    assert result["grounded"] is False
    assert result["answer"] == NO_ANSWER
    assert result["sources"] == []


def test_generator_is_required_when_score_is_high():
    """Above threshold the bot must actually call the model, not silently abstain."""
    with tempfile.TemporaryDirectory() as tmp:
        _toy_store().save(tmp)
        bot = SupportChatbot(
            index_dir=tmp,
            embedder=StubEmbedder([1.0, 0.0, 0.0]),
            load_generator=False,
        )
        try:
            bot.answer("about x")
        except RuntimeError:
            return
    raise AssertionError("expected RuntimeError because no generator was loaded")


def test_retrieve_returns_ranked_sources():
    with tempfile.TemporaryDirectory() as tmp:
        _toy_store().save(tmp)
        bot = SupportChatbot(
            index_dir=tmp,
            embedder=StubEmbedder([1.0, 0.0, 0.0]),
            load_generator=False,
        )
        results = bot.retrieve("about x", top_k=2)
    assert results[0]["ticket_id"] == "A"
    assert results[0]["score"] >= results[1]["score"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
