"""
A small local vector store with the same shape as the Pinecone index used in
production, so the pipeline can run with no API keys and no company data.

Vectors are L2-normalized on the way in, so cosine similarity is a dot product.
Thirty tickets do not need an ANN index, a brute-force matmul is exact and instant.

Requires:
    pip install numpy
"""

from __future__ import annotations

import json
import os

import numpy as np


class LocalVectorStore:
    """Brute-force cosine-similarity store over normalized vectors."""

    def __init__(self, vectors: np.ndarray | None = None, metadata: list[dict] | None = None):
        self.vectors = vectors if vectors is not None else np.zeros((0, 0), dtype=np.float32)
        self.metadata = metadata or []

    def upsert(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        """Add vectors and their metadata to the store."""
        if len(vectors) != len(metadata):
            raise ValueError(
                f"vector count {len(vectors)} does not match metadata count {len(metadata)}"
            )
        vectors = np.asarray(vectors, dtype=np.float32)
        if self.vectors.size == 0:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.metadata.extend(metadata)

    def query(self, vector: np.ndarray, top_k: int = 3) -> list[dict]:
        """Return the top_k most similar entries, each with its score."""
        if self.vectors.size == 0:
            return []

        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm

        scores = self.vectors @ vector
        top_k = min(top_k, len(scores))
        # argpartition finds the top_k cheaply, then we sort just those.
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]

        return [{"score": float(scores[i]), **self.metadata[i]} for i in idx]

    def save(self, directory: str) -> None:
        """Persist vectors and metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "vectors.npy"), self.vectors)
        with open(os.path.join(directory, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(self.metadata)} vectors to {directory}")

    @classmethod
    def load(cls, directory: str) -> "LocalVectorStore":
        """Load a store previously written by save()."""
        vectors_path = os.path.join(directory, "vectors.npy")
        metadata_path = os.path.join(directory, "metadata.json")
        if not os.path.exists(vectors_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"No index found in {directory}. Run build_index.py first."
            )
        vectors = np.load(vectors_path)
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        return cls(vectors=vectors, metadata=metadata)

    def __len__(self) -> int:
        return len(self.metadata)
