"""
Pydantic models for the Concord fact knowledge layer.

These define the API contracts and internal data shapes.
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
    UNIT_DIFFERENCE = "unit_difference"
    DEFINITION_DIFFERENCE = "definition_difference"
    NONE = "none"


class MatchSource(str, Enum):
    """How a candidate pair was discovered."""

    STRUCTURAL = "structural"
    EMBEDDING = "embedding"
    BOTH = "both"


class KeyType(str, Enum):
    """Types of normalization registry entries."""

    SUBJECT = "subject"
    ATTRIBUTE = "attribute"


# --- Page Chunk (PDF parsing output) ---


class PageChunk(BaseModel):
    """A single page of text extracted from a PDF, with optional table data."""

    doc_id: str
    doc_name: str
    page_number: int
    text: str
    tables: Optional[list[list[list[Optional[str]]]]] = None  # PyMuPDF table data
    table_row_count: int = 0


# --- Fact ---


class ExtractedFact(BaseModel):
    """
    A fact as extracted by the LLM, before normalization registry resolution.
    This is the raw LLM output shape.
    """

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
    created_at: Optional[datetime] = None


# --- Candidate Pair (intermediate, before judging) ---


class CandidatePair(BaseModel):
    """A pair of facts identified as potentially related, pending LLM judgment."""

    fact_1: Fact
    fact_2: Fact
    match_source: MatchSource
    match_hint: Optional[str] = None  # e.g. "exact_scope", "different_scope"
    is_intra_document: bool = False


# --- Normalization Registry ---


class NormalizationEntry(BaseModel):
    """A canonical normalization key in the registry."""

    id: Optional[str] = None
    key_type: KeyType
    raw_value: str
    canonical_value: str
    created_at: Optional[datetime] = None


# --- API Request / Response Models ---


class DocumentInfo(BaseModel):
    """Metadata about a processed document."""

    doc_id: str
    doc_name: str
    page_count: int
    fact_count: int
    created_at: Optional[datetime] = None


class FactDetail(BaseModel):
    """A fact with its relationships for the detail view."""

    fact: Fact
    relationships: list[Relationship] = []
    related_facts: list[Fact] = []


class ProcessingStatus(BaseModel):
    """Status update during PDF processing."""

    doc_id: str
    doc_name: str
    stage: str
    detail: str
    progress: Optional[float] = None  # 0.0 to 1.0


class ValidateKeyRequest(BaseModel):
    """Request to validate an LLM API key."""

    api_key: str
    model: Optional[str] = None


class ValidateKeyResponse(BaseModel):
    """Response from key validation."""

    valid: bool
    message: str
    model: Optional[str] = None
