"""
Hybrid candidate matching module.

Lane 1 (Primary): Structural key matching via (subject_normalized, attribute_normalized)
                  and claim_fingerprint. Fast, deterministic, and maps directly
                  to corroboration, contradiction, and reconciliation cases.
Lane 2 (Fallback): Vector embedding similarity search via sentence-transformers.
                   Catches synonyms, paraphrases, and implicit relationships that
                   differ in phrasing.
"""

from __future__ import annotations

import logging
from typing import Optional

from backend.config import EMBEDDING_SIMILARITY_THRESHOLD, EMBEDDING_TOP_K
from backend.database import Database
from backend.embeddings import EmbeddingService
from backend.models import CandidatePair, Fact, MatchSource

logger = logging.getLogger(__name__)


def make_pair_key(id_a: str, id_b: str) -> tuple[str, str]:
    """Ensure canonical pair order so (A, B) and (B, A) share the same key."""
    return (id_a, id_b) if id_a < id_b else (id_b, id_a)


class CandidateMatcher:
    """
    Two-lane hybrid candidate matcher for finding related facts across documents.
    """

    def __init__(
        self,
        db: Database,
        embedding_service: Optional[EmbeddingService] = None,
        embedding_threshold: float = EMBEDDING_SIMILARITY_THRESHOLD,
        top_k: int = EMBEDDING_TOP_K,
    ):
        self.db = db
        self.embedding_service = embedding_service or EmbeddingService()
        self.embedding_threshold = embedding_threshold
        self.top_k = top_k

    async def find_candidates(
        self,
        new_facts: list[Fact],
        existing_facts: Optional[list[Fact]] = None,
        check_existing_relationships: bool = True,
    ) -> list[CandidatePair]:
        """
        Find candidate fact pairs for the relation judge using two-lane matching.

        Args:
            new_facts: Newly extracted or targeted facts to find matches for.
            existing_facts: Optional existing facts to compare against. If None,
                            loads from the database.
            check_existing_relationships: If True, skips pairs that already have
                                          a judged relationship in the database.

        Returns:
            List of CandidatePair models ready for the relation judge.
        """
        if not new_facts:
            return []

        # Map to track unique candidates by canonical (id1, id2) key
        candidate_map: dict[tuple[str, str], CandidatePair] = {}

        # 1. Load comparison facts
        if existing_facts is None:
            all_db_facts = await self.db.get_facts(limit=3000)
            comparison_facts = [Fact(**f) for f in all_db_facts]
        else:
            comparison_facts = existing_facts

        # Index comparison facts by ID for fast lookup
        fact_by_id: dict[str, Fact] = {f.id: f for f in comparison_facts if f.id}
        for f in new_facts:
            if f.id:
                fact_by_id[f.id] = f

        # --- LANE 1: Structural Key Matching (Deterministic) ---
        for new_fact in new_facts:
            if not new_fact.id:
                continue

            # Find facts with same normalized subject and attribute
            struct_matches = [
                f
                for f in comparison_facts
                if f.id != new_fact.id
                and f.subject_normalized == new_fact.subject_normalized
                and f.attribute_normalized == new_fact.attribute_normalized
            ]

            for matched_fact in struct_matches:
                if not matched_fact.id:
                    continue

                # Nuance Check: Sibling decomposition check
                # Sibling facts decomposed from the same sentence share extraction_group_id
                if (
                    new_fact.extraction_group_id
                    and matched_fact.extraction_group_id
                    and new_fact.extraction_group_id == matched_fact.extraction_group_id
                ):
                    continue

                pair_key = make_pair_key(new_fact.id, matched_fact.id)

                # Incremental Ingestion: skip if already judged
                if check_existing_relationships and await self.db.relationship_exists(
                    new_fact.id, matched_fact.id
                ):
                    continue

                # Classify match hint based on claim fingerprint
                if new_fact.claim_fingerprint == matched_fact.claim_fingerprint:
                    hint = "exact_scope"  # Likely corroboration or contradiction
                else:
                    hint = "different_scope"  # Likely reconciliation

                # Order facts deterministically in pair
                f1 = fact_by_id[pair_key[0]]
                f2 = fact_by_id[pair_key[1]]

                candidate_map[pair_key] = CandidatePair(
                    fact_1=f1,
                    fact_2=f2,
                    match_source=MatchSource.STRUCTURAL,
                    match_hint=hint,
                    is_intra_document=(f1.source_doc_id == f2.source_doc_id),
                )

        # --- LANE 2: Vector Embedding Similarity (Semantic Fallback) ---
        try:
            # Ensure new facts have embeddings generated and stored
            new_fact_embeddings: dict[str, any] = {}
            for new_fact in new_facts:
                if not new_fact.id:
                    continue
                emb = self.embedding_service.embed_fact(new_fact)
                new_fact_embeddings[new_fact.id] = emb
                # Persist embedding to DB if connected
                try:
                    await self.db.update_fact_embedding(
                        new_fact.id, self.embedding_service.to_bytes(emb)
                    )
                except Exception as e:
                    logger.debug("Could not update embedding in DB: %s", e)

            # Get all stored embeddings from DB for comparison
            all_stored_embeddings = await self.db.get_all_fact_embeddings()

            for new_fact in new_facts:
                if not new_fact.id:
                    continue
                query_vec = new_fact_embeddings.get(new_fact.id)
                if query_vec is None:
                    continue

                # Filter out self
                candidates_for_query = [
                    (fid, vec)
                    for (fid, vec) in all_stored_embeddings
                    if fid != new_fact.id
                ]

                top_matches = self.embedding_service.find_top_k(
                    query_vec=query_vec,
                    candidate_embeddings=candidates_for_query,
                    top_k=self.top_k,
                    threshold=self.embedding_threshold,
                )

                for other_id, score in top_matches:
                    other_fact = fact_by_id.get(other_id)
                    if not other_fact:
                        continue

                    # Sibling decomposition check
                    if (
                        new_fact.extraction_group_id
                        and other_fact.extraction_group_id
                        and new_fact.extraction_group_id == other_fact.extraction_group_id
                    ):
                        continue

                    pair_key = make_pair_key(new_fact.id, other_id)

                    # Incremental Ingestion: skip if already judged
                    if check_existing_relationships and await self.db.relationship_exists(
                        new_fact.id, other_id
                    ):
                        continue

                    f1 = fact_by_id[pair_key[0]]
                    f2 = fact_by_id[pair_key[1]]

                    if pair_key in candidate_map:
                        # Already discovered via structural matching -> Upgrade to BOTH
                        candidate_map[pair_key].match_source = MatchSource.BOTH
                    else:
                        # Discovered solely via embedding lane
                        candidate_map[pair_key] = CandidatePair(
                            fact_1=f1,
                            fact_2=f2,
                            match_source=MatchSource.EMBEDDING,
                            match_hint=f"semantic_similarity_{score:.2f}",
                            is_intra_document=(f1.source_doc_id == f2.source_doc_id),
                        )
        except Exception as e:
            logger.warning("Lane 2 embedding matching encountered error: %s", e)

        candidates = list(candidate_map.values())
        logger.info(
            "Found %d candidate pairs (%d structural, %d embedding, %d both)",
            len(candidates),
            sum(1 for c in candidates if c.match_source == MatchSource.STRUCTURAL),
            sum(1 for c in candidates if c.match_source == MatchSource.EMBEDDING),
            sum(1 for c in candidates if c.match_source == MatchSource.BOTH),
        )
        return candidates


def match_facts_in_memory(
    facts: list[Fact],
    embedding_service: Optional[EmbeddingService] = None,
    embedding_threshold: float = EMBEDDING_SIMILARITY_THRESHOLD,
) -> list[CandidatePair]:
    """
    Pure in-memory candidate matching for testing and lightweight pipelines.
    """
    if len(facts) < 2:
        return []

    candidate_map: dict[tuple[str, str], CandidatePair] = {}
    fact_by_id = {f.id: f for f in facts if f.id}

    # Lane 1: Structural matching
    for i in range(len(facts)):
        for j in range(i + 1, len(facts)):
            f1 = facts[i]
            f2 = facts[j]
            if not f1.id or not f2.id:
                continue

            # Sibling check
            if (
                f1.extraction_group_id
                and f2.extraction_group_id
                and f1.extraction_group_id == f2.extraction_group_id
            ):
                continue

            # Structural key match
            if (
                f1.subject_normalized == f2.subject_normalized
                and f1.attribute_normalized == f2.attribute_normalized
            ):
                hint = (
                    "exact_scope"
                    if f1.claim_fingerprint == f2.claim_fingerprint
                    else "different_scope"
                )
                pair_key = make_pair_key(f1.id, f2.id)
                candidate_map[pair_key] = CandidatePair(
                    fact_1=fact_by_id[pair_key[0]],
                    fact_2=fact_by_id[pair_key[1]],
                    match_source=MatchSource.STRUCTURAL,
                    match_hint=hint,
                    is_intra_document=(f1.source_doc_id == f2.source_doc_id),
                )

    # Lane 2: Embedding matching
    if embedding_service:
        embeddings = embedding_service.embed_facts(facts)
        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                f1 = facts[i]
                f2 = facts[j]
                if not f1.id or not f2.id:
                    continue

                if (
                    f1.extraction_group_id
                    and f2.extraction_group_id
                    and f1.extraction_group_id == f2.extraction_group_id
                ):
                    continue

                pair_key = make_pair_key(f1.id, f2.id)
                sim = embedding_service.cosine_similarity(embeddings[i], embeddings[j])

                if sim >= embedding_threshold:
                    if pair_key in candidate_map:
                        candidate_map[pair_key].match_source = MatchSource.BOTH
                    else:
                        candidate_map[pair_key] = CandidatePair(
                            fact_1=fact_by_id[pair_key[0]],
                            fact_2=fact_by_id[pair_key[1]],
                            match_source=MatchSource.EMBEDDING,
                            match_hint=f"semantic_similarity_{sim:.2f}",
                            is_intra_document=(f1.source_doc_id == f2.source_doc_id),
                        )

    return list(candidate_map.values())
