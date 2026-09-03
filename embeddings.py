"""
Shared sentence-embedding helper used by both the Pinecone upload path and the
local retrieval path.

This replaces the pooling that pinecone_script.py did inline. The original took a
plain torch.mean over last_hidden_state after padding every input to max_length,
which averaged roughly 480 padding vectors into every short ticket and diluted the
signal. all-MiniLM-L6-v2 expects attention-mask-weighted mean pooling, which is what
mean_pool below implements.

Requires:
    pip install torch transformers numpy
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
MAX_TOKENS = 512


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average token vectors, ignoring padding positions.

    last_hidden_state: (batch, tokens, hidden)
    attention_mask:    (batch, tokens) with 1 for real tokens and 0 for padding
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class Embedder:
    """Wraps the tokenizer and model so callers do not repeat pooling logic."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        print(f"Loading embedding model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        print("Embedding model ready.")

    def encode(self, texts: list[str], batch_size: int = 16, normalize: bool = True) -> np.ndarray:
        """Embed a list of texts into a (len(texts), 384) float32 array."""
        if isinstance(texts, str):
            raise TypeError("encode expects a list of strings, not a single string")

        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tokens = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_TOKENS,
            )
            with torch.no_grad():
                output = self.model(**tokens)
            pooled = mean_pool(output.last_hidden_state, tokens["attention_mask"])
            vectors.append(pooled.cpu().numpy())

        result = np.vstack(vectors).astype(np.float32)
        if normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = result / np.clip(norms, 1e-9, None)
        return result

    def encode_one(self, text: str, normalize: bool = True) -> np.ndarray:
        """Embed a single string into a (384,) float32 array."""
        return self.encode([text], normalize=normalize)[0]


def ticket_text(ticket: dict) -> str:
    """Build the text that represents a ticket for embedding.

    Kept in one place so the index and any later re-index use identical wording.
    """
    return f"Subject: {ticket['Subject']}. Messages: {ticket['Messages']}"
