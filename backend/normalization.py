"""
Normalization registry for entity and attribute resolution.

Maintains canonical lookup table of normalized keys across documents.
Uses fuzzy matching (Levenshtein / token sort ratio via `thefuzz`) to ensure
LLM normalization consistency across documents processed at different times.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from thefuzz import fuzz

from backend.config import NORMALIZATION_SIMILARITY_THRESHOLD
from backend.database import Database

logger = logging.getLogger(__name__)


def clean_snake_case(text: str) -> str:
    """Convert text to clean lowercase snake_case without punctuation."""
    if not text:
        return "unspecified"
    # Replace non-alphanumeric characters with underscore
    cleaned = re.sub(r"[^\w\s-]", "", text.strip())
    cleaned = re.sub(r"[\s-]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_").lower()
    return cleaned or "unspecified"


def generate_claim_fingerprint(
    subject_normalized: str,
    attribute_normalized: str,
    temporal_scope: Optional[str] = None,
) -> str:
    """
    Generate a structural claim fingerprint.

    Format: `{subject_norm}::{attribute_norm}::{temporal_scope or 'unspecified'}`
    Facts with identical fingerprints are automatic candidates for corroboration or contradiction.
    Facts with matching prefix (subject + attribute) but different temporal scopes
    are prime reconciliation candidates.
    """
    subj = clean_snake_case(subject_normalized)
    attr = clean_snake_case(attribute_normalized)
    scope = clean_snake_case(temporal_scope) if temporal_scope else "unspecified"
    return f"{subj}::{attr}::{scope}"


class NormalizationRegistry:
    """
    Entity & Attribute Resolution Registry.

    When the LLM proposes a normalized key (e.g., 'delhivery_limited'),
    this registry checks previously stored canonical keys. If a fuzzy match
    exceeds the similarity threshold, the existing canonical key is returned,
    ensuring deterministic structural matching across documents.
    """

    def __init__(self, db: Database, threshold: float = NORMALIZATION_SIMILARITY_THRESHOLD):
        self.db = db
        self.threshold = threshold
        # Local cache: key_type -> set of canonical strings
        self._cache: dict[str, set[str]] = {"subject": set(), "attribute": set()}
        self._initialized = False

    async def initialize(self) -> None:
        """Pre-populate the local cache from the database."""
        try:
            for key_type in ("subject", "attribute"):
                rows = await self.db.get_canonical_values(key_type)
                self._cache[key_type] = {r["canonical_value"] for r in rows}
            self._initialized = True
            logger.debug(
                "NormalizationRegistry initialized with %d subjects, %d attributes",
                len(self._cache["subject"]),
                len(self._cache["attribute"]),
            )
        except Exception as e:
            logger.warning("Could not pre-load normalization cache: %s", e)

    async def resolve(
        self,
        key_type: str,
        raw_value: str,
        proposed_normalized: str,
    ) -> str:
        """
        Resolve a proposed normalized key to an existing canonical key or register a new one.

        Args:
            key_type: 'subject' or 'attribute'
            raw_value: Original text from document (e.g., 'Delhivery Limited')
            proposed_normalized: LLM proposed snake_case key (e.g., 'delhivery_ltd')

        Returns:
            Resolved canonical normalized key (e.g., 'delhivery')
        """
        normalized_proposal = clean_snake_case(proposed_normalized)
        if not normalized_proposal or normalized_proposal == "unspecified":
            normalized_proposal = clean_snake_case(raw_value)

        # Standard corporate suffix cleanups for subjects
        if key_type == "subject":
            normalized_proposal = self._clean_subject_suffixes(normalized_proposal)

        # Ensure cache is ready
        if not self._initialized:
            await self.initialize()

        cached_canonicals = self._cache.get(key_type, set())

        # Exact match in cache
        if normalized_proposal in cached_canonicals:
            return normalized_proposal

        # Fuzzy matching against existing canonicals
        best_match: Optional[str] = None
        best_score = 0

        for canonical in cached_canonicals:
            # token_sort_ratio handles word order differences
            # (e.g., 'reliance_industries' vs 'industries_reliance')
            score = fuzz.token_sort_ratio(normalized_proposal, canonical)
            if score > best_score:
                best_score = score
                best_match = canonical

        # If similarity exceeds threshold, reuse canonical
        if best_match and best_score >= self.threshold:
            logger.debug(
                "Fuzzy match resolved '%s' -> '%s' (score=%d, threshold=%d)",
                normalized_proposal,
                best_match,
                best_score,
                int(self.threshold),
            )
            # Record the raw alias in DB for tracking
            try:
                await self.db.insert_normalization_entry(
                    key_type=key_type,
                    raw_value=raw_value,
                    canonical_value=best_match,
                )
            except Exception as e:
                logger.warning("Error logging alias to registry: %s", e)

            return best_match

        # Otherwise, register the new normalized key as a new canonical value
        try:
            await self.db.insert_normalization_entry(
                key_type=key_type,
                raw_value=raw_value,
                canonical_value=normalized_proposal,
            )
            self._cache.setdefault(key_type, set()).add(normalized_proposal)
            logger.debug(
                "Registered new canonical %s: '%s' (from '%s')",
                key_type,
                normalized_proposal,
                raw_value,
            )
        except Exception as e:
            logger.warning("Error saving new canonical to registry: %s", e)

        return normalized_proposal

    @staticmethod
    def _clean_subject_suffixes(text: str) -> str:
        """Strip corporate legal suffixes while keeping the core entity name."""
        suffixes = (
            "_limited",
            "_ltd",
            "_pvt_ltd",
            "_private_limited",
            "_inc",
            "_incorporated",
            "_corp",
            "_corporation",
            "_llc",
            "_co",
            "_sa",
            "_ag",
        )
        for s in suffixes:
            if text.endswith(s) and len(text) > len(s):
                return text[: -len(s)].rstrip("_")
        return text
