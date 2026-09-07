"""
Smoke tests for the Concord pipeline.

Tests will be expanded as each phase is implemented.
"""

import pytest


def test_imports():
    """Verify all backend modules can be imported."""
    from backend import config
    from backend import models
    from backend import database


def test_config_paths():
    """Verify config paths are relative to project root, not hardcoded."""
    from backend.config import PROJECT_ROOT, UPLOAD_DIR, DB_PATH

    assert PROJECT_ROOT.exists()
    assert str(UPLOAD_DIR).startswith(str(PROJECT_ROOT))
    assert str(DB_PATH).startswith(str(PROJECT_ROOT))


def test_models():
    """Verify Pydantic models instantiate correctly."""
    from backend.models import ExtractedFact, Fact, Relationship, RelationType

    fact = ExtractedFact(
        subject="Test Corp",
        subject_normalized="test_corp",
        attribute="revenue",
        attribute_normalized="revenue",
        value="100",
        unit="million USD",
        temporal_scope="FY2023",
        evidence_quote="Revenue was 100 million USD in FY2023.",
        page=1,
        confidence=0.9,
    )
    assert fact.subject_normalized == "test_corp"
    assert fact.confidence == 0.9


def test_pdf_parser():
    """Verify PDF parser extracts text with correct page numbers."""
    from pathlib import Path

    from backend.config import PROJECT_ROOT
    from backend.pdf_parser import parse_pdf, get_page_count

    # Use the smallest starter PDF for testing
    pdf_path = (
        PROJECT_ROOT
        / "starter-datasets"
        / "delhivery"
        / "03-delhivery-q4-fy24-earnings-presentation.pdf"
    )
    assert pdf_path.exists(), f"Test PDF not found at {pdf_path}"

    file_bytes = pdf_path.read_bytes()

    # Test page count
    page_count = get_page_count(file_bytes)
    assert page_count == 27

    # Test full parsing
    chunks = parse_pdf(file_bytes, doc_name="test.pdf", doc_id="test-123")
    assert len(chunks) > 0
    assert all(c.doc_id == "test-123" for c in chunks)
    assert all(c.doc_name == "test.pdf" for c in chunks)
    assert all(c.page_number >= 1 for c in chunks)
    assert all(len(c.text) > 0 for c in chunks)

    # Page numbers should be 1-based and sequential (for pages with text)
    page_numbers = [c.page_number for c in chunks]
    assert page_numbers == sorted(page_numbers)


def test_prompts_structure():
    """Verify extraction and relation judge prompts contain all required guidelines."""
    from backend.prompts import (
        FACT_EXTRACTION_SYSTEM_PROMPT,
        FACT_EXTRACTION_USER_PROMPT_TEMPLATE,
        RELATION_JUDGE_SYSTEM_PROMPT,
    )

    # Check key requirements in extraction prompt
    assert "evidence_quote" in FACT_EXTRACTION_SYSTEM_PROMPT
    assert "VERBATIM" in FACT_EXTRACTION_SYSTEM_PROMPT
    assert "extraction_group_id" in FACT_EXTRACTION_SYSTEM_PROMPT
    assert "subject_normalized" in FACT_EXTRACTION_SYSTEM_PROMPT
    assert "attribute_normalized" in FACT_EXTRACTION_SYSTEM_PROMPT
    assert "temporal_scope" in FACT_EXTRACTION_SYSTEM_PROMPT

    # Check prompt template parameters
    assert "{doc_name}" in FACT_EXTRACTION_USER_PROMPT_TEMPLATE
    assert "{page_number}" in FACT_EXTRACTION_USER_PROMPT_TEMPLATE
    assert "{page_text}" in FACT_EXTRACTION_USER_PROMPT_TEMPLATE

    # Check relation judge prompt
    assert "corroborates" in RELATION_JUDGE_SYSTEM_PROMPT
    assert "contradicts" in RELATION_JUDGE_SYSTEM_PROMPT
    assert "reconciled" in RELATION_JUDGE_SYSTEM_PROMPT
    assert "temporal_scope" in RELATION_JUDGE_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_normalization_registry_fuzzy_matching(tmp_path):
    """Verify NormalizationRegistry resolves fuzzy keys and preserves canonical entities."""
    from backend.database import Database
    from backend.normalization import (
        NormalizationRegistry,
        clean_snake_case,
        generate_claim_fingerprint,
    )

    test_db_path = tmp_path / "test_norm.db"
    db = Database(str(test_db_path))
    await db.connect()

    try:
        registry = NormalizationRegistry(db, threshold=80.0)
        await registry.initialize()

        # 1. First entity registered
        c1 = await registry.resolve(
            key_type="subject",
            raw_value="Delhivery Limited",
            proposed_normalized="delhivery_limited",
        )
        assert c1 == "delhivery"  # Corporate suffix '_limited' cleaned

        # 2. Similar entity from another document (e.g. 'Delhivery Ltd')
        c2 = await registry.resolve(
            key_type="subject",
            raw_value="Delhivery Ltd",
            proposed_normalized="delhivery_ltd",
        )
        # Should fuzzy match and resolve to the same canonical 'delhivery'
        assert c2 == "delhivery"

        # 3. Different company should NOT match
        c3 = await registry.resolve(
            key_type="subject",
            raw_value="Reserve Bank of India",
            proposed_normalized="reserve_bank_of_india",
        )
        assert c3 == "reserve_bank_of_india"

        # 4. Attribute normalization fuzzy matching
        a1 = await registry.resolve(
            key_type="attribute",
            raw_value="Revenue from Operations",
            proposed_normalized="revenue_operations",
        )
        assert a1 == "revenue_operations"

        a2 = await registry.resolve(
            key_type="attribute",
            raw_value="Total Revenue from Operations",
            proposed_normalized="revenue_from_operations",
        )
        # 'revenue_operations' vs 'revenue_from_operations' token sort ratio > 80%
        assert a2 == "revenue_operations"

        # 5. Claim fingerprint generation
        fp1 = generate_claim_fingerprint("delhivery", "revenue_operations", "Q4_FY24")
        assert fp1 == "delhivery::revenue_operations::q4_fy24"

        fp2 = generate_claim_fingerprint("delhivery", "revenue_operations", None)
        assert fp2 == "delhivery::revenue_operations::unspecified"
    finally:
        await db.close()


def test_parse_llm_facts_and_decomposition():
    """Verify parsing of LLM JSON responses with compound fact decomposition."""
    from backend.fact_extractor import parse_llm_facts

    sample_llm_response = """
    Here are the extracted facts:
    ```json
    {
      "facts": [
        {
          "subject": "Delhivery",
          "subject_normalized": "delhivery",
          "attribute": "Revenue from operations",
          "attribute_normalized": "revenue_operations",
          "value": "2075.54",
          "unit": "INR Crore",
          "temporal_scope": "Q4_FY24",
          "conditions": "consolidated",
          "evidence_quote": "Revenue from operations for Q4 FY24 stood at Rs. 2,075.54 Cr",
          "page": 5,
          "confidence": 0.95,
          "extraction_group_id": "group_rev_1"
        },
        {
          "subject": "Delhivery",
          "subject_normalized": "delhivery",
          "attribute": "Revenue from operations",
          "attribute_normalized": "revenue_operations",
          "value": "1850.20",
          "unit": "INR Crore",
          "temporal_scope": "Q4_FY23",
          "conditions": "consolidated",
          "evidence_quote": "compared to Rs. 1,850.20 Cr in Q4 FY23",
          "page": 5,
          "confidence": 0.95,
          "extraction_group_id": "group_rev_1"
        }
      ]
    }
    ```
    """

    facts = parse_llm_facts(sample_llm_response, default_page=5)
    assert len(facts) == 2
    assert facts[0].subject == "Delhivery"
    assert facts[0].value == "2075.54"
    assert facts[0].temporal_scope == "Q4_FY24"
    assert facts[0].extraction_group_id == "group_rev_1"
    assert facts[1].value == "1850.20"
    assert facts[1].temporal_scope == "Q4_FY23"
    # Sibling facts share the same extraction_group_id
    assert facts[0].extraction_group_id == facts[1].extraction_group_id


@pytest.mark.asyncio
async def test_process_and_store_facts_pipeline(tmp_path):
    """Verify processing and persisting extracted facts into database."""
    from backend.database import Database
    from backend.fact_extractor import process_and_store_facts
    from backend.models import ExtractedFact
    from backend.normalization import NormalizationRegistry

    test_db_path = tmp_path / "test_store.db"
    db = Database(str(test_db_path))
    await db.connect()

    try:
        await db.insert_document("doc-1", "annual_report.pdf", 10)
        registry = NormalizationRegistry(db)
        await registry.initialize()

        extracted = [
            ExtractedFact(
                subject="Delhivery Limited",
                subject_normalized="delhivery_limited",
                attribute="Revenue from operations",
                attribute_normalized="revenue_operations",
                value="2075.54",
                unit="INR Crore",
                temporal_scope="Q4_FY24",
                evidence_quote="Revenue from operations for Q4 FY24 stood at Rs. 2,075.54 Cr",
                page=3,
                confidence=0.92,
                extraction_group_id="g1",
            )
        ]

        saved_facts = await process_and_store_facts(
            extracted_facts=extracted,
            doc_id="doc-1",
            doc_name="annual_report.pdf",
            db=db,
            registry=registry,
        )

        assert len(saved_facts) == 1
        fact = saved_facts[0]
        assert fact.id is not None
        assert fact.source_doc_id == "doc-1"
        assert fact.subject_normalized == "delhivery"
        assert fact.attribute_normalized == "revenue_operations"
        assert fact.claim_fingerprint == "delhivery::revenue_operations::q4_fy24"
        assert fact.extraction_group_id == "doc-1_3_g1"

        # Verify DB query
        db_facts = await db.get_facts(source_doc_id="doc-1")
        assert len(db_facts) == 1
        assert db_facts[0]["claim_fingerprint"] == "delhivery::revenue_operations::q4_fy24"
    finally:
        await db.close()

