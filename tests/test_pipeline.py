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


def test_embedding_service_operations():
    """Verify embedding generation, vector serialization, and cosine similarity."""
    import numpy as np

    from backend.config import EMBEDDING_DIMENSION
    from backend.embeddings import EmbeddingService, format_fact_for_embedding
    from backend.models import Fact

    service = EmbeddingService()

    fact1 = Fact(
        id="f1",
        subject="Delhivery",
        subject_normalized="delhivery",
        attribute="Revenue",
        attribute_normalized="revenue",
        value="2075.54",
        unit="INR Cr",
        temporal_scope="Q4_FY24",
        claim_fingerprint="delhivery::revenue::q4_fy24",
        source_doc="doc1.pdf",
        source_doc_id="d1",
        page=1,
        evidence_quote="Revenue was 2075.54 INR Cr",
    )

    formatted = format_fact_for_embedding(fact1)
    assert "delhivery" in formatted
    assert "2075.54" in formatted

    vec1 = service.embed_fact(fact1)
    assert isinstance(vec1, np.ndarray)
    assert vec1.shape == (EMBEDDING_DIMENSION,)
    assert vec1.dtype == np.float32

    # Serialization test
    blob = service.to_bytes(vec1)
    recovered = service.from_bytes(blob)
    assert np.allclose(vec1, recovered)

    # Cosine similarity: self similarity should be ~1.0
    self_sim = service.cosine_similarity(vec1, vec1)
    assert abs(self_sim - 1.0) < 1e-4


def test_candidate_matching_structural_and_siblings():
    """Verify Lane 1 structural matching classifies hints and excludes sibling facts."""
    from backend.matcher import match_facts_in_memory
    from backend.models import Fact, MatchSource

    # Fact 1: Delhivery Q4 revenue in Doc 1
    f1 = Fact(
        id="fact-1",
        subject="Delhivery",
        subject_normalized="delhivery",
        attribute="Revenue from operations",
        attribute_normalized="revenue_operations",
        value="2075.54",
        unit="INR Crore",
        temporal_scope="Q4_FY24",
        claim_fingerprint="delhivery::revenue_operations::q4_fy24",
        extraction_group_id="group_a",
        source_doc="earnings_doc.pdf",
        source_doc_id="doc_a",
        page=2,
        evidence_quote="Q4 revenue stood at 2075.54 Cr",
    )

    # Fact 2: Sibling of Fact 1 from same sentence (e.g. YoY growth)
    f2_sibling = Fact(
        id="fact-2",
        subject="Delhivery",
        subject_normalized="delhivery",
        attribute="Revenue from operations",
        attribute_normalized="revenue_operations",
        value="12",
        unit="%",
        temporal_scope="Q4_FY24",
        claim_fingerprint="delhivery::revenue_operations::q4_fy24",
        extraction_group_id="group_a",  # SAME GROUP AS F1
        source_doc="earnings_doc.pdf",
        source_doc_id="doc_a",
        page=2,
        evidence_quote="representing 12% growth",
    )

    # Fact 3: Same company, same metric, same period in Doc 2 (Corroboration candidate)
    f3 = Fact(
        id="fact-3",
        subject="Delhivery Limited",
        subject_normalized="delhivery",
        attribute="Revenue from operations",
        attribute_normalized="revenue_operations",
        value="2075",
        unit="INR Crore",
        temporal_scope="Q4_FY24",
        claim_fingerprint="delhivery::revenue_operations::q4_fy24",
        extraction_group_id="group_b",
        source_doc="press_release.pdf",
        source_doc_id="doc_b",
        page=1,
        evidence_quote="Fourth quarter revenue reached approximately Rs 2075 Cr",
    )

    # Fact 4: Same company & metric, DIFFERENT period (Reconciliation candidate)
    f4 = Fact(
        id="fact-4",
        subject="Delhivery",
        subject_normalized="delhivery",
        attribute="Revenue from operations",
        attribute_normalized="revenue_operations",
        value="1850.20",
        unit="INR Crore",
        temporal_scope="Q4_FY23",
        claim_fingerprint="delhivery::revenue_operations::q4_fy23",
        extraction_group_id="group_c",
        source_doc="annual_report.pdf",
        source_doc_id="doc_c",
        page=4,
        evidence_quote="Previous year Q4 revenue was 1850.20 Cr",
    )

    candidates = match_facts_in_memory([f1, f2_sibling, f3, f4])

    # Sibling pair (f1, f2_sibling) MUST NOT be matched
    sibling_pairs = [
        c
        for c in candidates
        if (c.fact_1.id == "fact-1" and c.fact_2.id == "fact-2")
        or (c.fact_1.id == "fact-2" and c.fact_2.id == "fact-1")
    ]
    assert len(sibling_pairs) == 0, "Sibling facts from same sentence should not be paired"

    # Exact scope match (f1 and f3)
    exact_pairs = [
        c
        for c in candidates
        if {c.fact_1.id, c.fact_2.id} == {"fact-1", "fact-3"}
    ]
    assert len(exact_pairs) == 1
    assert exact_pairs[0].match_hint == "exact_scope"
    assert exact_pairs[0].match_source == MatchSource.STRUCTURAL
    assert exact_pairs[0].is_intra_document is False

    # Different scope match (f1 and f4)
    diff_pairs = [
        c
        for c in candidates
        if {c.fact_1.id, c.fact_2.id} == {"fact-1", "fact-4"}
    ]
    assert len(diff_pairs) == 1
    assert diff_pairs[0].match_hint == "different_scope"


@pytest.mark.asyncio
async def test_candidate_matcher_hybrid_with_db(tmp_path):
    """Verify CandidateMatcher with async SQLite database and both matching lanes."""
    from backend.database import Database
    from backend.embeddings import EmbeddingService
    from backend.matcher import CandidateMatcher
    from backend.models import Fact, MatchSource

    test_db_path = tmp_path / "test_matcher.db"
    db = Database(str(test_db_path))
    await db.connect()

    try:
        embedding_service = EmbeddingService()
        matcher = CandidateMatcher(
            db=db,
            embedding_service=embedding_service,
            embedding_threshold=0.65,
        )

        # Existing fact in Doc A
        f1 = Fact(
            id="f-101",
            subject="Delhivery",
            subject_normalized="delhivery",
            attribute="Revenue",
            attribute_normalized="revenue",
            value="2075.54",
            unit="INR Cr",
            temporal_scope="Q4_FY24",
            claim_fingerprint="delhivery::revenue::q4_fy24",
            source_doc="doc_a.pdf",
            source_doc_id="doc-a",
            page=1,
            evidence_quote="Revenue was 2075.54 INR Cr",
        )
        await db.insert_fact(f1.model_dump())

        # New fact from Doc B
        f2 = Fact(
            id="f-102",
            subject="Delhivery",
            subject_normalized="delhivery",
            attribute="Revenue",
            attribute_normalized="revenue",
            value="2075",
            unit="INR Cr",
            temporal_scope="Q4_FY24",
            claim_fingerprint="delhivery::revenue::q4_fy24",
            source_doc="doc_b.pdf",
            source_doc_id="doc-b",
            page=2,
            evidence_quote="Delhivery posted 2075 INR Cr revenue",
        )
        await db.insert_fact(f2.model_dump())

        # Find candidates for f2
        candidates = await matcher.find_candidates(
            new_facts=[f2],
            check_existing_relationships=True,
        )

        assert len(candidates) >= 1
        pair = candidates[0]
        assert {pair.fact_1.id, pair.fact_2.id} == {"f-101", "f-102"}
        assert pair.match_hint == "exact_scope"
        assert pair.is_intra_document is False
        # Discovered via both structural and embedding similarity
        assert pair.match_source in (MatchSource.STRUCTURAL, MatchSource.BOTH)
    finally:
        await db.close()


def test_relation_judge_parsing_and_normalization():
    """Verify parsing and normalization of relation judge LLM responses."""
    from backend.models import CandidatePair, Fact, MatchSource, ReconcilingFactor, RelationType
    from backend.relation_judge import (
        extract_json_from_llm,
        normalize_reconciling_factor,
        normalize_relation_type,
        parse_judgment_dict,
    )

    f1 = Fact(
        id="f1",
        subject="Delhivery",
        subject_normalized="delhivery",
        attribute="Revenue",
        attribute_normalized="revenue",
        value="2075.54",
        unit="INR Cr",
        temporal_scope="Q4_FY24",
        claim_fingerprint="delhivery::revenue::q4_fy24",
        source_doc="doc1.pdf",
        source_doc_id="d1",
        page=1,
        evidence_quote="Revenue was 2075.54 Cr",
    )
    f2 = Fact(
        id="f2",
        subject="Delhivery",
        subject_normalized="delhivery",
        attribute="Revenue",
        attribute_normalized="revenue",
        value="1850.20",
        unit="INR Cr",
        temporal_scope="Q4_FY23",
        claim_fingerprint="delhivery::revenue::q4_fy23",
        source_doc="doc2.pdf",
        source_doc_id="d2",
        page=3,
        evidence_quote="Revenue was 1850.20 Cr in Q4 FY23",
    )
    pair = CandidatePair(
        fact_1=f1,
        fact_2=f2,
        match_source=MatchSource.STRUCTURAL,
        match_hint="different_scope",
        is_intra_document=False,
    )

    # 1. Test normalization helpers
    assert normalize_relation_type("CORROBORATION") == RelationType.CORROBORATES
    assert normalize_relation_type("contradiction") == RelationType.CONTRADICTS
    assert normalize_relation_type("reconcile") == RelationType.RECONCILED

    assert normalize_reconciling_factor("time_difference", RelationType.RECONCILED) == ReconcilingFactor.TEMPORAL_SCOPE
    assert normalize_reconciling_factor("unit_scale", RelationType.RECONCILED) == ReconcilingFactor.UNIT_DIFFERENCE
    assert normalize_reconciling_factor("subsidiary_scope", RelationType.RECONCILED) == ReconcilingFactor.ENTITY_SCOPE
    assert normalize_reconciling_factor("temporal_scope", RelationType.CORROBORATES) == ReconcilingFactor.NONE

    # 2. Test JSON extraction from markdown fences
    llm_markdown = """
    ```json
    {
      "relation_type": "reconciled",
      "reconciling_factor": "temporal_scope",
      "explanation": "Values differ because Fact 1 describes Q4 FY24 (2075.54 Cr) while Fact 2 describes Q4 FY23 (1850.20 Cr)."
    }
    ```
    """
    parsed_json = extract_json_from_llm(llm_markdown)
    relationship = parse_judgment_dict(parsed_json, pair)

    assert relationship.relation_type == RelationType.RECONCILED
    assert relationship.reconciling_factor == ReconcilingFactor.TEMPORAL_SCOPE
    assert "Q4 FY24" in relationship.explanation
    assert relationship.fact_id_1 == "f1"
    assert relationship.fact_id_2 == "f2"


@pytest.mark.asyncio
async def test_judge_and_store_candidates_db_pipeline(tmp_path):
    """Verify persisting judged relationships and demo cases in SQLite."""
    from backend.database import Database
    from backend.models import CandidatePair, Fact, MatchSource, ReconcilingFactor, RelationType

    test_db_path = tmp_path / "test_rel.db"
    db = Database(str(test_db_path))
    await db.connect()

    try:
        # Insert test facts for the 3 core relationship cases
        f1 = Fact(
            id="f-c1",
            subject="Delhivery",
            subject_normalized="delhivery",
            attribute="Revenue",
            attribute_normalized="revenue",
            value="2075.54",
            claim_fingerprint="delhivery::revenue::q4_fy24",
            source_doc="doc1.pdf",
            source_doc_id="d1",
            page=1,
            evidence_quote="Revenue: 2075.54 Cr",
        )
        f2 = Fact(
            id="f-c2",
            subject="Delhivery",
            subject_normalized="delhivery",
            attribute="Revenue",
            attribute_normalized="revenue",
            value="2075",
            claim_fingerprint="delhivery::revenue::q4_fy24",
            source_doc="doc2.pdf",
            source_doc_id="d2",
            page=2,
            evidence_quote="Posted 2075 Cr revenue",
        )
        f3 = Fact(
            id="f-c3",
            subject="Delhivery",
            subject_normalized="delhivery",
            attribute="Revenue",
            attribute_normalized="revenue",
            value="1850.20",
            claim_fingerprint="delhivery::revenue::q4_fy23",
            source_doc="doc1.pdf",
            source_doc_id="d1",
            page=1,
            evidence_quote="Previous year Q4 was 1850.20 Cr",
        )

        await db.insert_fact(f1.model_dump())
        await db.insert_fact(f2.model_dump())
        await db.insert_fact(f3.model_dump())

        # Store Corroboration relationship
        rel_corr = {
            "fact_id_1": "f-c1",
            "fact_id_2": "f-c2",
            "relation_type": RelationType.CORROBORATES.value,
            "reconciling_factor": ReconcilingFactor.NONE.value,
            "explanation": "Both documents state approximately 2,075 Cr revenue for Q4 FY24.",
            "match_source": MatchSource.STRUCTURAL.value,
            "is_intra_document": 0,
        }
        await db.insert_relationship(rel_corr)

        # Store Reconciled relationship
        rel_rec = {
            "fact_id_1": "f-c1",
            "fact_id_2": "f-c3",
            "relation_type": RelationType.RECONCILED.value,
            "reconciling_factor": ReconcilingFactor.TEMPORAL_SCOPE.value,
            "explanation": "Different values represent different fiscal quarters (Q4 FY24 vs Q4 FY23).",
            "match_source": MatchSource.STRUCTURAL.value,
            "is_intra_document": 1,
        }
        await db.insert_relationship(rel_rec)

        # Verify DB queries
        all_rels = await db.get_relationships()
        assert len(all_rels) == 2

        corr_rels = await db.get_relationships(relation_type="corroborates")
        assert len(corr_rels) == 1
        assert corr_rels[0]["fact_id_1"] == "f-c1"

        rec_rels = await db.get_relationships(relation_type="reconciled")
        assert len(rec_rels) == 1
        assert rec_rels[0]["reconciling_factor"] == "temporal_scope"

        # Verify relationship_exists check works symmetrically
        assert await db.relationship_exists("f-c1", "f-c2") is True
        assert await db.relationship_exists("f-c2", "f-c1") is True
        assert await db.relationship_exists("f-c2", "f-c3") is False
    finally:
        await db.close()


def test_rest_api_endpoints_comprehensive(monkeypatch):
    """Verify all REST API endpoints using FastAPI TestClient."""
    from fastapi.testclient import TestClient
    from backend.main import app

    monkeypatch.setattr("backend.main.get_effective_key", lambda *args, **kwargs: None)

    with TestClient(app) as client:
        # 1. Health check
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

        # 2. Upload a test PDF
        from backend.config import PROJECT_ROOT
        pdf_path = (
            PROJECT_ROOT
            / "starter-datasets"
            / "delhivery"
            / "03-delhivery-q4-fy24-earnings-presentation.pdf"
        )
        with open(pdf_path, "rb") as f:
            upload_resp = client.post("/api/upload", files={"file": ("delhivery_q4.pdf", f, "application/pdf")})
        assert upload_resp.status_code == 200
        doc_data = upload_resp.json()
        doc_id = doc_data["doc_id"]
        assert doc_data["chunks_extracted"] == 27
        assert doc_data["status"] == "parsed_only"

        # 3. Documents list
        r = client.get("/api/documents")
        assert r.status_code == 200
        docs = r.json()["documents"]
        assert any(d["doc_id"] == doc_id for d in docs)

        # 4. Single document metadata
        r = client.get(f"/api/documents/{doc_id}")
        assert r.status_code == 200
        assert r.json()["document"]["doc_id"] == doc_id
        assert "fact_count" in r.json()["document"]

        # 5. Document facts
        r = client.get(f"/api/documents/{doc_id}/facts")
        assert r.status_code == 200
        assert "facts" in r.json()

        # 6. Document relationships
        r = client.get(f"/api/documents/{doc_id}/relationships")
        assert r.status_code == 200
        assert "relationships" in r.json()

        # 7. Facts endpoint with filtering
        r = client.get("/api/facts", params={"limit": 50})
        assert r.status_code == 200

        r = client.get("/api/facts", params={"search": "delhivery"})
        assert r.status_code == 200

        # 8. Relationships endpoint with filtering
        r = client.get("/api/relationships", params={"relation_type": "corroborates"})
        assert r.status_code == 200

        # 9. Normalization canonicals
        r = client.get("/api/normalization/canonicals")
        assert r.status_code == 200
        assert "subjects" in r.json()
        assert "attributes" in r.json()

        # 10. Export endpoint
        r = client.get("/api/export")
        assert r.status_code == 200
        export_data = r.json()
        assert "export_metadata" in export_data
        assert "documents" in export_data
        assert "facts" in export_data
        assert "relationships" in export_data

        # 11. Delete document
        r = client.delete(f"/api/documents/{doc_id}")
        assert r.status_code == 200
        assert r.json()["status"] == "deleted"

        # Verify 404 after delete
        r = client.get(f"/api/documents/{doc_id}")
        assert r.status_code == 404




