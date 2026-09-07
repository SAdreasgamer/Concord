"""
Embedding service using sentence-transformers.

Generates local semantic vector embeddings for facts and calculates cosine
similarities for Lane 2 candidate matching. Completely local and self-contained —
no external API key or network calls required for embeddings.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

import numpy as np

from backend.config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_SIMILARITY_THRESHOLD,
    EMBEDDING_TOP_K,
)
from backend.models import Fact

logger = logging.getLogger(__name__)


def format_fact_for_embedding(fact: Union[Fact, dict[str, Any]]) -> str:
    """
    Format a fact into a clean semantic string representation for embedding.

    Example: "delhivery: revenue operations is 2075.54 INR Crore (Q4_FY24) consolidated"
    """
    if isinstance(fact, Fact):
        subject = fact.subject_normalized or fact.subject
        attribute = fact.attribute_normalized or fact.attribute
        value = fact.value
        unit = fact.unit or ""
        temporal = fact.temporal_scope or ""
        conditions = fact.conditions or ""
    else:
        subject = fact.get("subject_normalized") or fact.get("subject", "")
        attribute = fact.get("attribute_normalized") or fact.get("attribute", "")
        value = str(fact.get("value", ""))
        unit = fact.get("unit") or ""
        temporal = fact.get("temporal_scope") or ""
        conditions = fact.get("conditions") or ""

    parts = [f"{subject}: {attribute} is {value}"]
    if unit:
        parts.append(unit)
    if temporal and temporal != "unspecified":
        parts.append(f"({temporal})")
    if conditions:
        parts.append(f"[{conditions}]")

    return " ".join(parts).strip()


class EmbeddingService:
    """
    Service for generating and comparing vector embeddings.

    Uses lazy model loading so that the heavy sentence-transformer model
    is only loaded when embedding operations are actually requested.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        self._model = None

    @property
    def model(self):
        """Lazy load the sentence-transformer model."""
        if self._model is None:
            logger.info("Loading sentence-transformers model '%s'...", self.model_name)
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            logger.info("Model '%s' loaded successfully.", self.model_name)
        return self._model

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string into a unit-normalized float32 vector.
        """
        vec = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype(np.float32)

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        """
        Batch embed multiple text strings into unit-normalized vectors.
        """
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return [v.astype(np.float32) for v in vectors]

    def embed_fact(self, fact: Union[Fact, dict[str, Any]]) -> np.ndarray:
        """Generate a vector embedding for a single fact."""
        text = format_fact_for_embedding(fact)
        return self.embed_text(text)

    def embed_facts(
        self, facts: list[Union[Fact, dict[str, Any]]]
    ) -> list[np.ndarray]:
        """Batch generate embeddings for multiple facts."""
        texts = [format_fact_for_embedding(f) for f in facts]
        return self.embed_texts(texts)

    @staticmethod
    def to_bytes(vector: np.ndarray) -> bytes:
        """Serialize float32 numpy vector to raw bytes for SQLite BLOB storage."""
        return vector.astype(np.float32).tobytes()

    @staticmethod
    def from_bytes(blob: bytes) -> np.ndarray:
        """Deserialize raw bytes from SQLite BLOB to float32 numpy vector."""
        return np.frombuffer(blob, dtype=np.float32)

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two unit-normalized vectors.
        For unit vectors, cosine similarity is simply the dot product.
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def find_top_k(
        self,
        query_vec: np.ndarray,
        candidate_embeddings: list[tuple[str, np.ndarray]],
        top_k: int = EMBEDDING_TOP_K,
        threshold: float = EMBEDDING_SIMILARITY_THRESHOLD,
    ) -> list[tuple[str, float]]:
        """
        Find top-k most similar fact IDs above the similarity threshold.

        Args:
            query_vec: The query embedding (1D numpy array).
            candidate_embeddings: List of (fact_id, embedding_vector) tuples.
            top_k: Maximum number of matches to return.
            threshold: Minimum cosine similarity required.

        Returns:
            List of (fact_id, similarity_score) sorted descending by similarity.
        """
        if not candidate_embeddings:
            return []

        # Vectorized similarity calculation
        fact_ids = [item[0] for item in candidate_embeddings]
        matrix = np.vstack([item[1] for item in candidate_embeddings])

        # Normalize query vector if needed
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            q_unit = query_vec / q_norm
        else:
            return []

        # Normalize candidate matrix rows
        m_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        m_norms[m_norms == 0] = 1.0
        matrix_unit = matrix / m_norms

        # Dot products give cosine similarities
        scores = np.dot(matrix_unit, q_unit)

        results = []
        for fact_id, score in zip(fact_ids, scores):
            score_val = float(score)
            if score_val >= threshold:
                results.append((fact_id, score_val))

        # Sort descending by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
