"""
Pydantic models for Concord.

Defines API contracts and internal data shapes.
The fact schema is intentionally flexible — most fields are optional/nullable
so new document types don't require schema changes.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --- Enums ---


class RelationType(str, Enum):
    """Types of relationships between facts."""
    CORROBORATES = "corroborates"
    CONTRADICTS = "contradicts"
    RECONCILED = "reconciled"


class ReconcilingFactor(str, Enum):
    """Factors that can reconcile an apparent contradiction."""
    TEMPORAL_SCOPE = "temporal_scope"
    ENTITY_SCOPE = "entity_scope"
    REPORTING_BASIS = "reporting_basis"
    PROJECTION_VS_ACTUAL = "projection_vs_actual"
    DEFINITION_DIFFERENCE = "definition_difference"
    PRECISION_OR_ROUNDING = "precision_or_rounding"
    UNIT_DIFFERENCE = "unit_difference"
    NONE = "none"


class MatchSource(str, Enum):
    """How a candidate pair was discovered."""
    STRUCTURAL = "structural"
    FUZZY = "fuzzy"


# --- Page Chunk (PDF parsing output) ---


class PageChunk(BaseModel):
    """A single page of text extracted from a PDF, with optional table data and density metrics."""
    doc_id: str
    doc_name: str
    page_number: int
    text: str
    tables: Optional[list[list[list[Optional[str]]]]] = None
    table_row_count: int = 0
    density_score: float = 0.0
    is_front_matter: bool = False


# --- Fact ---


class ExtractedFact(BaseModel):
    """A fact as extracted by the LLM, before fingerprinting and storage."""
    subject: str
    subject_normalized: str
    attribute: str
    attribute_normalized: str
    value: str
    unit: Optional[str] = None
    temporal_scope: Optional[str] = None
    conditions: Optional[str] = None
    evidence_quote: str
    page: int
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    extraction_group_id: Optional[str] = None


class Fact(BaseModel):
    """A fully resolved fact stored in the database."""
    id: Optional[str] = None
    subject: str
    subject_normalized: str
    attribute: str
    attribute_normalized: str
    value: str
    unit: Optional[str] = None
    temporal_scope: Optional[str] = None
    conditions: Optional[str] = None
    claim_fingerprint: str
    extraction_group_id: Optional[str] = None
    source_doc: str
    source_doc_id: str
    page: int
    evidence_quote: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    created_at: Optional[datetime] = None


# --- Relationship ---


class Relationship(BaseModel):
    """A judged relationship between two facts."""
    id: Optional[str] = None
    fact_id_1: str
    fact_id_2: str
    relation_type: RelationType
    reconciling_factor: ReconcilingFactor = ReconcilingFactor.NONE
    explanation: str
    match_source: MatchSource
    is_intra_document: bool = False
    agreement_strength: float = 1.0
    created_at: Optional[datetime] = None


# --- Candidate Pair (intermediate, before judging) ---


class CandidatePair(BaseModel):
    """A pair of facts identified as potentially related, pending judgment."""
    fact_1: Fact
    fact_2: Fact
    match_source: MatchSource
    match_hint: Optional[str] = None
    is_intra_document: bool = False
