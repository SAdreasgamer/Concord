"""
Comprehensive integration and unit test suite for Concord.
Tests all current modules: config, models, database, parser, prompts, extractor, matcher, judge, and REST API.
Runs completely offline without external network calls or LLM dependencies.
"""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from backend.config import PROJECT_ROOT, UPLOAD_DIR, DB_PATH, DEFAULT_PAGE_LIMIT, DEFAULT_LLM_MODEL
from backend.models import ExtractedFact, Fact, CandidatePair, Relationship, RelationType, ReconcilingFactor, MatchSource
from backend.database import Database
from backend.pdf_parser import parse_pdf, get_page_count, calculate_page_density, get_high_signal_chunks, PageChunk
from backend.prompts import FACT_EXTRACTION_SYSTEM_PROMPT, FACT_EXTRACTION_USER_PROMPT_TEMPLATE, RELATION_JUDGE_SYSTEM_PROMPT
from backend.fact_extractor import parse_llm_facts, repair_and_parse_json, clean_snake_case, generate_claim_fingerprint
from backend.matcher import are_attributes_related, find_candidates
from backend.relation_judge import _extract_json, _normalize_relation_type, _normalize_factor


def test_imports_and_config():
    """Verify backend modules import cleanly and config paths are properly rooted."""
    assert PROJECT_ROOT.exists()
    assert str(UPLOAD_DIR).startswith(str(PROJECT_ROOT))
    assert str(DB_PATH).startswith(str(PROJECT_ROOT))
    assert DEFAULT_PAGE_LIMIT == 15
    assert DEFAULT_LLM_MODEL is not None


def test_models_instantiation():
    """Verify Pydantic models instantiate and validate correctly."""
    fact = ExtractedFact(
        subject="Delhivery",
        subject_normalized="delhivery",
        attribute="Net worth",
        attribute_normalized="net_worth",
        value="59798.47",
        unit="₹ million",
        temporal_scope="FY2021",
        evidence_quote="Net worth: 59798.47",
        page=5,
        confidence=0.95,
    )
    assert fact.subject == "Delhivery"
    assert fact.confidence == 0.95

    rel = Relationship(
        id="rel-123",
        fact_id_1="fact-1",
        fact_id_2="fact-2",
        relation_type=RelationType.RECONCILED,
        reconciling_factor=ReconcilingFactor.TEMPORAL_SCOPE,
        explanation="Values reflect different reporting years.",
        match_source=MatchSource.FUZZY,
        agreement_strength=0.95,
    )
    assert rel.relation_type == RelationType.RECONCILED
    assert rel.reconciling_factor == ReconcilingFactor.TEMPORAL_SCOPE


def test_pdf_parser_and_density_scoring():
    """Verify PDF parser, page density scoring, and high-signal selection."""
    test_pdf = PROJECT_ROOT / "starter-datasets" / "delhivery" / "03-delhivery-q4-fy24-earnings-presentation.pdf"
    assert test_pdf.exists()

    file_bytes = test_pdf.read_bytes()
    page_count = get_page_count(file_bytes)
    assert page_count == 27

    # Parse first 5 pages
    chunks = parse_pdf(file_bytes, doc_name="test.pdf", doc_id="doc-123", max_pages=5)
    assert len(chunks) == 5
    assert chunks[0].page_number == 1
    assert chunks[-1].page_number == 5

    # Density calculation
    sample_text = "Revenue in FY24 grew by 15.5% to reach 8,141 Cr compared to 7,224 Cr in FY23."
    score, is_fm = calculate_page_density(sample_text)
    assert score > 0.0

    # High signal selection
    top_chunks = get_high_signal_chunks(chunks, max_chunks=2)
    assert len(top_chunks) <= 2


def test_prompts_and_templates():
    """Verify system prompts are concise, grounded, and require evidence quotes."""
    assert "evidence_quote" in FACT_EXTRACTION_SYSTEM_PROMPT
    assert "VERBATIM" in FACT_EXTRACTION_SYSTEM_PROMPT
    assert "{doc_name}" in FACT_EXTRACTION_USER_PROMPT_TEMPLATE
    assert "{page_number}" in FACT_EXTRACTION_USER_PROMPT_TEMPLATE
    assert "{page_text}" in FACT_EXTRACTION_USER_PROMPT_TEMPLATE

    assert "corroborates" in RELATION_JUDGE_SYSTEM_PROMPT
    assert "contradicts" in RELATION_JUDGE_SYSTEM_PROMPT
    assert "reconciled" in RELATION_JUDGE_SYSTEM_PROMPT


def test_fact_extraction_parsing_and_repair():
    """Verify parsing LLM JSON with robust repair and decomposition."""
    raw_response = """
    ```json
    {
      "facts": [
        {
          "subject": "Delhivery",
          "attribute": "Revenue from operations",
          "value": "81415",
          "unit": "₹ million",
          "temporal_scope": "FY24",
          "evidence_quote": "Revenue from operations: 81415",
          "confidence": 0.95
        }
      ]
    }
    ```
    """
    facts = parse_llm_facts(raw_response, default_page=4)
    assert len(facts) == 1
    assert facts[0].subject == "Delhivery"
    assert facts[0].value == "81415"
    assert facts[0].temporal_scope == "FY24"
    assert facts[0].page == 4

    # Broken/clipped JSON repair
    clipped_json = '{"facts": [{"subject": "RBI", "attribute": "GDP growth", "value": "8.2%", "evidence_quote": "GDP was 8.2%"}]}'
    parsed = repair_and_parse_json(clipped_json)
    assert parsed is not None
    assert "facts" in parsed
    assert parsed["facts"][0]["subject"] == "RBI"


def test_clean_snake_case_and_fingerprints():
    """Verify normalization and deterministic claim fingerprint generation."""
    assert clean_snake_case("Delhivery Limited") == "delhivery"
    assert clean_snake_case("Revenue from Operations") == "revenue_from_operations"

    fp1 = generate_claim_fingerprint("delhivery", "net_worth", "Dec 31, 2021")
    assert fp1 == "delhivery::net_worth::cy2021"

    fp2 = generate_claim_fingerprint("delhivery", "revenue", "FY2024")
    assert fp2 == "delhivery::revenue::fy2024"


def test_candidate_matching_domain_agnostic():
    """Verify domain-agnostic semantic word overlap matching."""
    # Related attributes
    assert are_attributes_related("Revenue from operations", "Total revenue") is True
    assert are_attributes_related("EBITDA", "EBITDA margin") is True
    assert are_attributes_related("Net profit", "Profit after tax") is True

    # Unrelated attributes (should NOT match)
    assert are_attributes_related("Net worth", "Net working capital cycle") is False
    assert are_attributes_related("Fleet size", "Share price") is False


def test_relation_judge_parsing():
    """Verify relationship judgment parsing, normalization, and regex fallback."""
    raw_valid = '{"relation_type": "corroborates", "reconciling_factor": "none", "confidence": 0.95, "explanation": "Identical values."}'
    parsed = _extract_json(raw_valid)
    assert parsed["relation_type"] == "corroborates"
    assert parsed["reconciling_factor"] == "none"

    # Clipped LLM output fallback
    clipped = '{"relation_type": "reconciled", "reconciling_factor": "temporal_scope", "explanation": "Different fiscal periods'
    parsed_fallback = _extract_json(clipped)
    assert parsed_fallback["relation_type"] == "reconciled"
    assert parsed_fallback["reconciling_factor"] == "temporal_scope"

    # Normalization functions
    assert _normalize_relation_type("CORROBORATES") == RelationType.CORROBORATES
    assert _normalize_relation_type("contradictory") == RelationType.CONTRADICTS
    assert _normalize_factor("TEMPORAL_SCOPE", RelationType.RECONCILED) == ReconcilingFactor.TEMPORAL_SCOPE
    assert _normalize_factor("unknown_factor", RelationType.RECONCILED) == ReconcilingFactor.NONE



@pytest.mark.asyncio
async def test_database_lifecycle(tmp_path):
    """Verify database initialization, fact storage, querying, and cascade deletion."""
    test_db_file = tmp_path / "test_concord.db"
    db = Database(str(test_db_file))
    await db.connect()

    try:
        # 1. Insert document
        await db.insert_document("doc-1", "test_report.pdf", 10)
        doc = await db.get_document("doc-1")
        assert doc is not None
        assert doc["doc_name"] == "test_report.pdf"

        # 2. Insert fact
        fact_dict = {
            "id": "fact-1",
            "subject": "Delhivery",
            "subject_normalized": "delhivery",
            "attribute": "Revenue",
            "attribute_normalized": "revenue",
            "value": "81415",
            "unit": "₹ million",
            "temporal_scope": "FY24",
            "conditions": None,
            "claim_fingerprint": "delhivery::revenue::fy2024",
            "extraction_group_id": None,
            "source_doc": "test_report.pdf",
            "source_doc_id": "doc-1",
            "page": 4,
            "evidence_quote": "Revenue: 81415",
            "confidence": 0.95,
        }
        await db.insert_fact(fact_dict)
        facts = await db.get_facts(source_doc_id="doc-1")
        assert len(facts) == 1
        assert facts[0]["id"] == "fact-1"

        # 3. Insert relationship
        rel_dict = {
            "id": "rel-1",
            "fact_id_1": "fact-1",
            "fact_id_2": "fact-1",
            "relation_type": "corroborates",
            "reconciling_factor": "none",
            "explanation": "Self test",
            "match_source": "structural",
            "is_intra_document": 1,
            "agreement_strength": 0.95,
        }
        await db.insert_relationship(rel_dict)
        rels = await db.get_relationships()
        assert len(rels) == 1
        assert rels[0]["id"] == "rel-1"

        # 4. Cascade delete document
        await db.delete_document("doc-1")
        assert await db.get_document("doc-1") is None
        assert len(await db.get_facts(source_doc_id="doc-1")) == 0
        assert len(await db.get_relationships()) == 0
    finally:
        await db.close()


def test_fastapi_rest_endpoints():
    """Verify REST API endpoints with FastAPI TestClient."""
    from backend.main import app

    with TestClient(app) as client:
        # Health
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

        # Documents
        r = client.get("/api/documents")
        assert r.status_code == 200
        assert "documents" in r.json()

        # Facts
        r = client.get("/api/facts")
        assert r.status_code == 200
        assert "facts" in r.json()

        # Relationships
        r = client.get("/api/relationships")
        assert r.status_code == 200
        assert "relationships" in r.json()


        # Telemetry
        r = client.get("/api/telemetry")
        assert r.status_code == 200

