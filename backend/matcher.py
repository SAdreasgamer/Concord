"""
Candidate matching module.

Finds pairs of facts across documents that should be compared.
Two strategies:
  1. Structural: same subject_normalized + same attribute_normalized
  2. Fuzzy: same subject_normalized + attributes share meaningful word overlap
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from backend.database import Database
from backend.models import CandidatePair, Fact, MatchSource

logger = logging.getLogger(__name__)


# Words that carry no semantic signal for attribute matching
_STOP_WORDS = {
    "of", "in", "the", "and", "to", "for", "a", "an", "on", "as", "by", "at", "per", "from", "with",
    "average", "total", "net", "gross", "daily", "annual", "monthly", "quarterly",
    "number", "numbers", "share", "shares", "period", "year", "quarter", "value", "figure", "level", "rate", "ratio",
}


def _attribute_words(attr: str) -> set[str]:
    """Extract meaningful words from an attribute string."""
    return set(re.findall(r"[a-z0-9]+", attr.lower())) - _STOP_WORDS


def are_attributes_related(attr1: str, attr2: str) -> bool:
    """Check if two attributes are semantically related via word overlap in a domain-agnostic manner."""
    w1 = _attribute_words(attr1)
    w2 = _attribute_words(attr2)
    if not w1 or not w2:
        return False
    overlap = w1 & w2
    if not overlap:
        return False
    # If they share any meaningful content word of 4+ characters (e.g. 'revenue', 'ebitda', 'profit', 'growth')
    return any(len(w) >= 4 for w in overlap) or len(overlap) >= 2


def are_subjects_related(s1: str, s2: str) -> bool:
    """Check if two subjects refer to the same entity (exact, stem, or meaningful substring)."""
    if not s1 or not s2:
        return False
    if s1 == s2:
        return True
    if len(s1) >= 4 and len(s2) >= 4:
        if s1 in s2 or s2 in s1:
            return True
        words1 = set(s1.split("_"))
        words2 = set(s2.split("_"))
        overlap = words1 & words2
        if overlap and any(len(w) >= 4 for w in overlap):
            return True
    return False


def _pair_key(id_a: str, id_b: str) -> tuple[str, str]:

    """Canonical pair order so (A,B) and (B,A) share the same key."""
    return (id_a, id_b) if id_a < id_b else (id_b, id_a)


async def find_candidates(
    new_facts: list[Fact],
    db: Database,
    cross_document_only: bool = True,
) -> list[CandidatePair]:
    """
    Find candidate fact pairs for relationship judging.

    Compares new_facts against all existing facts in the database.
    Returns pairs that share the same subject and have related attributes.
    """
    if not new_facts:
        return []

    # Load all existing facts
    all_db_facts = await db.get_facts(limit=3000)
    existing_facts = [Fact(**f) for f in all_db_facts]

    # Index by ID
    fact_by_id: dict[str, Fact] = {f.id: f for f in existing_facts if f.id}
    for f in new_facts:
        if f.id:
            fact_by_id[f.id] = f

    candidate_map: dict[tuple[str, str], CandidatePair] = {}

    for new_fact in new_facts:
        if not new_fact.id:
            continue

        for existing in existing_facts:
            if not existing.id or existing.id == new_fact.id:
                continue

            # Skip intra-document pairs
            if cross_document_only and existing.source_doc_id == new_fact.source_doc_id:
                continue

            # Must be same or closely related entity
            if not are_subjects_related(existing.subject_normalized, new_fact.subject_normalized):
                continue


            # Skip sibling facts from same extraction group
            if (new_fact.extraction_group_id and existing.extraction_group_id
                    and new_fact.extraction_group_id == existing.extraction_group_id):
                continue

            # Check attribute relationship
            is_exact = existing.attribute_normalized == new_fact.attribute_normalized
            is_fuzzy = not is_exact and are_attributes_related(
                new_fact.attribute_normalized or new_fact.attribute,
                existing.attribute_normalized or existing.attribute,
            )

            if not is_exact and not is_fuzzy:
                continue

            pk = _pair_key(new_fact.id, existing.id)

            # Skip already judged
            if await db.relationship_exists(new_fact.id, existing.id):
                continue

            if pk in candidate_map:
                continue

            # Determine match hint
            if new_fact.claim_fingerprint == existing.claim_fingerprint:
                hint = "exact_scope"
            else:
                hint = "different_scope"

            f1 = fact_by_id[pk[0]]
            f2 = fact_by_id[pk[1]]

            candidate_map[pk] = CandidatePair(
                fact_1=f1,
                fact_2=f2,
                match_source=MatchSource.STRUCTURAL if is_exact else MatchSource.FUZZY,
                match_hint=hint,
                is_intra_document=(f1.source_doc_id == f2.source_doc_id),
            )

    candidates = list(candidate_map.values())
    # Prioritize exact structural matches over fuzzy matches, cap to top 4 to ensure instant execution
    candidates = sorted(candidates, key=lambda c: 0 if c.match_source == MatchSource.STRUCTURAL else 1)[:4]
    logger.info(
        "Selected top %d candidate pairs (%d structural, %d fuzzy)",
        len(candidates),
        sum(1 for c in candidates if c.match_source == MatchSource.STRUCTURAL),
        sum(1 for c in candidates if c.match_source == MatchSource.FUZZY),
    )
    return candidates

