"""
Concord — Fact Knowledge Layer API.

FastAPI application entry point. Serves the REST API and static frontend.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import PROJECT_ROOT
from backend.database import Database
from backend.embeddings import EmbeddingService
from backend.matcher import CandidateMatcher
from backend.normalization import NormalizationRegistry

# --- Singletons ---
db = Database()
registry = NormalizationRegistry(db)
embedding_service = EmbeddingService()
matcher = CandidateMatcher(db=db, embedding_service=embedding_service)


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    await db.connect()
    await registry.initialize()
    yield
    await db.close()


# --- App ---
app = FastAPI(
    title="Concord",
    description="Fact Knowledge Layer — extract, ground, and compare facts across PDF documents.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Health Check ---


@app.get("/health")
async def health():
    """Health check endpoint."""
    stats = await db.get_stats()
    return {"status": "ok", "stats": stats}


# --- Document Endpoints ---


@app.get("/api/documents")
async def list_documents():
    """List all processed documents with fact counts."""
    docs = await db.get_documents()
    return {"documents": docs}


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get metadata for a single document."""
    doc = await db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    facts = await db.get_facts(source_doc_id=doc_id)
    rels = await db.get_relationships_for_document(doc_id)
    return {
        "document": {
            **doc,
            "fact_count": len(facts),
            "relationship_count": len(rels),
        }
    }


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and cascade delete its facts and relationships."""
    success = await db.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted", "doc_id": doc_id}


@app.get("/api/documents/{doc_id}/facts")
async def get_document_facts(doc_id: str):
    """Get all facts from a specific document."""
    doc = await db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    facts = await db.get_facts(source_doc_id=doc_id)
    return {"document": doc, "facts": facts, "count": len(facts)}


@app.get("/api/documents/{doc_id}/relationships")
async def get_document_relationships(doc_id: str):
    """Get all relationships where at least one fact belongs to this document."""
    doc = await db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    relationships = await db.get_relationships_for_document(doc_id)
    enriched = []
    for rel in relationships:
        f1 = await db.get_fact(rel["fact_id_1"])
        f2 = await db.get_fact(rel["fact_id_2"])
        enriched.append({**rel, "fact_1": f1, "fact_2": f2})
    return {"document": doc, "relationships": enriched, "count": len(enriched)}


# --- Fact Endpoints ---


@app.get("/api/facts")
async def list_facts(
    source_doc_id: Optional[str] = Query(None),
    subject_normalized: Optional[str] = Query(None),
    attribute_normalized: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(500, ge=1, le=2000),
):
    """List all facts, with optional filters by document, entity, attribute, or search query."""
    all_facts = await db.get_facts(source_doc_id=source_doc_id, limit=limit)

    # In-memory filtering for fine-grained criteria
    filtered = all_facts
    if subject_normalized:
        s_norm = subject_normalized.strip().lower()
        filtered = [f for f in filtered if f["subject_normalized"] == s_norm]
    if attribute_normalized:
        a_norm = attribute_normalized.strip().lower()
        filtered = [f for f in filtered if f["attribute_normalized"] == a_norm]
    if search:
        q = search.strip().lower()
        filtered = [
            f
            for f in filtered
            if q in f["subject"].lower()
            or q in f["attribute"].lower()
            or q in f["value"].lower()
            or q in f["evidence_quote"].lower()
        ]

    return {"facts": filtered, "count": len(filtered)}


@app.get("/api/facts/{fact_id}")
async def get_fact_detail(fact_id: str):
    """Get a single fact with its relationships and related facts."""
    fact = await db.get_fact(fact_id)
    if not fact:
        raise HTTPException(status_code=404, detail="Fact not found")

    relationships = await db.get_relationships_for_fact(fact_id)

    # Collect all related fact IDs
    related_fact_ids = set()
    for rel in relationships:
        related_fact_ids.add(rel["fact_id_1"])
        related_fact_ids.add(rel["fact_id_2"])
    related_fact_ids.discard(fact_id)

    # Fetch related facts
    related_facts = []
    for rid in related_fact_ids:
        rf = await db.get_fact(rid)
        if rf:
            related_facts.append(rf)

    return {
        "fact": fact,
        "relationships": relationships,
        "related_facts": related_facts,
    }


@app.get("/api/facts/{fact_id}/related")
async def get_fact_related(fact_id: str):
    """Get all relationships for a specific fact, enriched with counterpart fact data."""
    fact = await db.get_fact(fact_id)
    if not fact:
        raise HTTPException(status_code=404, detail="Fact not found")

    relationships = await db.get_relationships_for_fact(fact_id)
    enriched = []
    for rel in relationships:
        counterpart_id = rel["fact_id_2"] if rel["fact_id_1"] == fact_id else rel["fact_id_1"]
        counterpart = await db.get_fact(counterpart_id)
        enriched.append({
            **rel,
            "target_fact": counterpart,
        })
    return {"fact_id": fact_id, "relationships": enriched, "count": len(enriched)}


# --- Relationship Endpoints ---


@app.get("/api/relationships")
async def list_relationships(
    relation_type: Optional[str] = Query(None),
    is_intra_document: Optional[bool] = Query(None),
    limit: int = Query(500, ge=1, le=2000),
):
    """List all relationships, optionally filtered by type or intra-document scope."""
    relationships = await db.get_relationships(
        relation_type=relation_type, limit=limit
    )

    if is_intra_document is not None:
        target_val = 1 if is_intra_document else 0
        relationships = [r for r in relationships if r.get("is_intra_document") == target_val]

    # Enrich with fact data
    enriched = []
    for rel in relationships:
        fact_1 = await db.get_fact(rel["fact_id_1"])
        fact_2 = await db.get_fact(rel["fact_id_2"])
        enriched.append({**rel, "fact_1": fact_1, "fact_2": fact_2})

    return {"relationships": enriched, "count": len(enriched)}


@app.get("/api/relationships/{rel_id}")
async def get_relationship_detail(rel_id: str):
    """Get a single relationship by ID with enriched fact details."""
    rel = await db.get_relationship(rel_id)
    if not rel:
        raise HTTPException(status_code=404, detail="Relationship not found")
    f1 = await db.get_fact(rel["fact_id_1"])
    f2 = await db.get_fact(rel["fact_id_2"])
    return {**rel, "fact_1": f1, "fact_2": f2}


# --- Export Endpoint ---


@app.get("/api/export")
async def export_data():
    """Export complete fact knowledge layer (documents, facts, relationships)."""
    docs = await db.get_documents()
    facts = await db.get_facts(limit=10000)
    rels = await db.get_relationships(limit=10000)
    stats = await db.get_stats()

    # Clean embeddings from export
    cleaned_facts = [{k: v for k, v in f.items() if k != "embedding"} for f in facts]

    return {
        "export_metadata": {
            "version": "0.1.0",
            "stats": stats,
        },
        "documents": docs,
        "facts": cleaned_facts,
        "relationships": rels,
    }


# --- Key Validation ---


def get_effective_key(header_key: Optional[str] = None, model: Optional[str] = None) -> Optional[str]:
    """Retrieve API key from request header or reload dynamically from .env file."""
    from dotenv import load_dotenv
    load_dotenv(override=True)
    if header_key and header_key.strip():
        return header_key.strip()

    if model:
        m = model.lower()
        if "groq" in m:
            return os.getenv("GROQ_API_KEY") or None
        if "gemini" in m:
            return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None
        if "gpt" in m or "openai" in m:
            return os.getenv("OPENAI_API_KEY") or None
        if "claude" in m or "anthropic" in m:
            return os.getenv("ANTHROPIC_API_KEY") or None

    return (
        os.getenv("GROQ_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or None
    )


@app.post("/api/validate-key")
async def validate_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    model: Optional[str] = Query(None),
):
    """Validate an LLM API key with a lightweight test call (supports X-API-Key header or .env)."""
    import litellm

    from backend.config import DEFAULT_LLM_MODEL

    test_model = model or DEFAULT_LLM_MODEL
    key_to_test = get_effective_key(x_api_key, model=test_model)

    if key_to_test and key_to_test.startswith("gsk_") and not test_model.startswith("groq/"):
        test_model = "groq/openai/gpt-oss-120b"

    if not key_to_test:
        return JSONResponse(
            status_code=400,
            content={
                "valid": False,
                "message": "No API key provided. Paste it in the UI or in the .env file.",
                "model": test_model,
            },
        )

    try:
        response = litellm.completion(
            model=test_model,
            messages=[{"role": "user", "content": "Reply with exactly: ok"}],
            api_key=key_to_test,
            max_tokens=50,
        )
        return {
            "valid": True,
            "message": "API key is valid",
            "model": test_model,
        }
    except Exception as e:
        return JSONResponse(
            status_code=401,
            content={
                "valid": False,
                "message": f"Key validation failed: {str(e)}",
                "model": test_model,
            },
        )


# --- Sample Datasets (One-Click Testing) ---

SAMPLE_DATASETS = {
    "delhivery_q4": {
        "title": "Delhivery Q4 FY24 Earnings",
        "category": "Earnings Presentation",
        "filename": "03-delhivery-q4-fy24-earnings-presentation.pdf",
        "rel_path": "starter-datasets/delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf",
    },
    "delhivery_prospectus": {
        "title": "Delhivery Prospectus 2022",
        "category": "IPO Prospectus Excerpt",
        "filename": "01-delhivery-prospectus-2022-excerpt.pdf",
        "rel_path": "starter-datasets/delhivery/01-delhivery-prospectus-2022-excerpt.pdf",
    },
    "delhivery_annual": {
        "title": "Delhivery Annual Report FY24",
        "category": "Annual Report Excerpt",
        "filename": "02-delhivery-annual-report-fy24-excerpt.pdf",
        "rel_path": "starter-datasets/delhivery/02-delhivery-annual-report-fy24-excerpt.pdf",
    },
    "economic_survey": {
        "title": "India Economic Survey 2024-25",
        "category": "Macroeconomy Excerpt",
        "filename": "01-india-economic-survey-2024-25-excerpt.pdf",
        "rel_path": "starter-datasets/india-macroeconomy/01-india-economic-survey-2024-25-excerpt.pdf",
    },
    "rbi_annual": {
        "title": "RBI Annual Report 2024-25",
        "category": "Central Bank Report",
        "filename": "02-rbi-annual-report-2024-25-excerpt.pdf",
        "rel_path": "starter-datasets/india-macroeconomy/02-rbi-annual-report-2024-25-excerpt.pdf",
    },
}


@app.get("/api/sample-datasets")
async def list_sample_datasets():
    """List bundled starter datasets available for instant one-click ingestion."""
    return {
        "samples": [
            {
                "key": k,
                "title": v["title"],
                "category": v["category"],
                "filename": v["filename"],
            }
            for k, v in SAMPLE_DATASETS.items()
        ]
    }


@app.post("/api/sample-datasets/{dataset_key}/load")
async def load_sample_dataset(
    dataset_key: str,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    model: Optional[str] = Query(None),
    max_pages: Optional[int] = Query(None),
):
    """
    Load and process a bundled starter dataset with a single click.
    Uses the hybrid extraction pipeline with telemetry.
    """
    import os
    import uuid

    from backend.config import PROJECT_ROOT, UPLOAD_DIR
    from backend.fact_extractor import extract_document_facts
    from backend.pdf_parser import get_page_count, parse_pdf
    from backend.telemetry import PipelineTelemetry

    sample = SAMPLE_DATASETS.get(dataset_key)
    if not sample:
        raise HTTPException(
            status_code=404,
            detail=f"Sample dataset '{dataset_key}' not found. Available: {list(SAMPLE_DATASETS.keys())}",
        )

    file_path = PROJECT_ROOT / sample["rel_path"]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Sample PDF file missing on disk")

    file_bytes = file_path.read_bytes()
    doc_id = str(uuid.uuid4())
    doc_name = sample["filename"]

    chunks = parse_pdf(file_bytes, doc_name=doc_name, doc_id=doc_id)
    page_count = get_page_count(file_bytes)

    # Save to uploads
    save_path = UPLOAD_DIR / f"{doc_id}.pdf"
    save_path.write_bytes(file_bytes)

    # Insert document
    await db.insert_document(doc_id, doc_name, page_count)

    from backend.config import DEFAULT_LLM_MODEL

    model = model or DEFAULT_LLM_MODEL
    effective_key = get_effective_key(x_api_key, model=model)
    if effective_key and effective_key.startswith("gsk_") and not model.startswith("groq/"):
        model = "groq/openai/gpt-oss-120b"

    facts_extracted = []
    relationships_found = []
    telemetry = PipelineTelemetry()
    telemetry.start()

    if effective_key:
        try:
            facts_extracted = await extract_document_facts(
                chunks=chunks,
                doc_id=doc_id,
                doc_name=doc_name,
                db=db,
                registry=registry,
                api_key=effective_key,
                model=model,
                max_pages=max_pages,
                telemetry=telemetry,
            )
            if facts_extracted:
                stage_match = telemetry.start_stage("matching_and_judging")
                candidates = await matcher.find_candidates(new_facts=facts_extracted)
                telemetry.candidate_pairs_found = len(candidates)
                if candidates:
                    from backend.relation_judge import judge_and_store_candidates

                    relationships_found = await judge_and_store_candidates(
                        candidates=candidates,
                        db=db,
                        api_key=effective_key,
                        model=model,
                    )
                    telemetry.relationships_discovered = len(relationships_found)
                    telemetry.judge_api_calls = len(candidates)
                stage_match.stop()
        except Exception as e:
            telemetry.stop()
            return {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "page_count": page_count,
                "chunks_extracted": len(chunks),
                "fact_count": 0,
                "relationship_count": 0,
                "status": "extraction_error",
                "message": f"Sample document parsed, but processing encountered: {str(e)}",
                "telemetry": telemetry.to_dict(),
            }

    telemetry.stop()
    from backend.telemetry import global_telemetry
    global_telemetry.record_run(doc_id, doc_name, telemetry)

    return {
        "doc_id": doc_id,
        "doc_name": doc_name,
        "page_count": page_count,
        "chunks_extracted": len(chunks),
        "fact_count": len(facts_extracted),
        "relationship_count": len(relationships_found),
        "status": "extracted" if effective_key else "parsed_only",
        "message": (
            f"Extracted {len(facts_extracted)} facts and discovered {len(relationships_found)} relationships."
            if effective_key
            else "Sample PDF parsed successfully. Provide API key to extract facts."
        ),
        "facts_preview": [f.model_dump() for f in facts_extracted[:15]],
        "relationships_preview": [r.model_dump() for r in relationships_found[:10]],
        "telemetry": telemetry.to_dict(),
    }


# --- PDF Upload & Processing ---


@app.post("/api/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    model: Optional[str] = Query(None),
    max_pages: Optional[int] = Query(None, description="Max pages to extract facts from"),
):
    """
    Upload a PDF file for processing.

    Uses the HYBRID extraction pipeline:
    1. Parse PDF → page text + table detection
    2. Classify pages → skip junk, route tables locally
    3. Batch remaining pages to LLM → fewer API calls
    4. Normalize → match → judge relationships
    """
    import os
    import uuid

    from backend.config import UPLOAD_DIR
    from backend.fact_extractor import extract_document_facts
    from backend.pdf_parser import get_page_count, parse_pdf
    from backend.telemetry import PipelineTelemetry

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted.",
        )

    # Read file content
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    doc_id = str(uuid.uuid4())
    doc_name = file.filename

    # Parse PDF into page chunks (now includes table detection)
    try:
        chunks = parse_pdf(file_bytes, doc_name=doc_name, doc_id=doc_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No extractable text found in the PDF. It may be a scanned document.",
        )

    # Save the uploaded file
    save_path = UPLOAD_DIR / f"{doc_id}.pdf"
    save_path.write_bytes(file_bytes)

    # Store document metadata in DB
    page_count = get_page_count(file_bytes)
    await db.insert_document(doc_id, doc_name, page_count)

    from backend.config import DEFAULT_LLM_MODEL

    model = model or DEFAULT_LLM_MODEL
    effective_key = get_effective_key(x_api_key, model=model)
    if effective_key and effective_key.startswith("gsk_") and not model.startswith("groq/"):
        model = "groq/openai/gpt-oss-120b"

    facts_extracted = []
    relationships_found = []
    telemetry = PipelineTelemetry()
    telemetry.start()

    if effective_key:
        try:
            # Stage: Hybrid extraction (classify → local tables → batch LLM)
            facts_extracted = await extract_document_facts(
                chunks=chunks,
                doc_id=doc_id,
                doc_name=doc_name,
                db=db,
                registry=registry,
                api_key=effective_key,
                model=model,
                max_pages=max_pages,
                telemetry=telemetry,
            )
            # Stage: Candidate matching and relation judging
            if facts_extracted:
                stage_match = telemetry.start_stage("matching_and_judging")
                candidates = await matcher.find_candidates(new_facts=facts_extracted)
                telemetry.candidate_pairs_found = len(candidates)
                if candidates:
                    from backend.relation_judge import judge_and_store_candidates

                    relationships_found = await judge_and_store_candidates(
                        candidates=candidates,
                        db=db,
                        api_key=effective_key,
                        model=model,
                    )
                    telemetry.relationships_discovered = len(relationships_found)
                    telemetry.judge_api_calls = len(candidates)  # Approximate
                stage_match.stop()
        except Exception as e:
            telemetry.stop()
            # Document is still saved even if LLM processing hits an error
            return {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "page_count": page_count,
                "chunks_extracted": len(chunks),
                "fact_count": 0,
                "relationship_count": 0,
                "status": "extraction_error",
                "message": f"Document parsed but processing failed: {str(e)}",
                "telemetry": telemetry.to_dict(),
            }

    telemetry.stop()
    from backend.telemetry import global_telemetry
    global_telemetry.record_run(doc_id, doc_name, telemetry)

    return {
        "doc_id": doc_id,
        "doc_name": doc_name,
        "page_count": page_count,
        "chunks_extracted": len(chunks),
        "fact_count": len(facts_extracted),
        "relationship_count": len(relationships_found),
        "status": "extracted" if effective_key else "parsed_only",
        "message": (
            f"Extracted {len(facts_extracted)} facts and discovered {len(relationships_found)} relationships."
            if effective_key
            else "Document parsed. Pass X-API-Key header to extract facts and judge relationships."
        ),
        "facts_preview": [f.model_dump() for f in facts_extracted[:15]],
        "relationships_preview": [r.model_dump() for r in relationships_found[:10]],
        "telemetry": telemetry.to_dict(),
    }


@app.post("/api/documents/{doc_id}/process")
async def process_document(
    doc_id: str,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    model: Optional[str] = Query(None),
    max_pages: Optional[int] = Query(None),
):
    """
    Extract facts and judge relationships from an existing document using an API key.
    Uses the hybrid extraction pipeline with telemetry.
    """
    import os

    from backend.config import UPLOAD_DIR
    from backend.fact_extractor import extract_document_facts
    from backend.pdf_parser import parse_pdf
    from backend.relation_judge import judge_and_store_candidates
    from backend.telemetry import PipelineTelemetry

    doc = await db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = UPLOAD_DIR / f"{doc_id}.pdf"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Source PDF file not found on server")

    from backend.config import DEFAULT_LLM_MODEL

    model = model or DEFAULT_LLM_MODEL
    effective_key = get_effective_key(x_api_key, model=model)
    if effective_key and effective_key.startswith("gsk_") and not model.startswith("groq/"):
        model = "groq/openai/gpt-oss-120b"

    if not effective_key:
        raise HTTPException(
            status_code=400,
            detail="API key required. Provide via X-API-Key header or set GROQ_API_KEY in .env.",
        )

    telemetry = PipelineTelemetry()
    telemetry.start()

    file_bytes = file_path.read_bytes()
    chunks = parse_pdf(file_bytes, doc_name=doc["doc_name"], doc_id=doc_id)

    facts = await extract_document_facts(
        chunks=chunks,
        doc_id=doc_id,
        doc_name=doc["doc_name"],
        db=db,
        registry=registry,
        api_key=effective_key,
        model=model,
        max_pages=max_pages,
        telemetry=telemetry,
    )

    relationships = []
    if facts:
        stage_match = telemetry.start_stage("matching_and_judging")
        candidates = await matcher.find_candidates(new_facts=facts)
        telemetry.candidate_pairs_found = len(candidates)
        if candidates:
            relationships = await judge_and_store_candidates(
                candidates=candidates,
                db=db,
                api_key=effective_key,
                model=model,
            )
            telemetry.relationships_discovered = len(relationships)
            telemetry.judge_api_calls = len(candidates)
        stage_match.stop()

    telemetry.stop()
    from backend.telemetry import global_telemetry
    global_telemetry.record_run(doc_id, doc.get("doc_name", doc_id), telemetry)

    return {
        "doc_id": doc_id,
        "doc_name": doc["doc_name"],
        "fact_count": len(facts),
        "relationship_count": len(relationships),
        "facts": [f.model_dump() for f in facts],
        "relationships": [r.model_dump() for r in relationships],
        "telemetry": telemetry.to_dict(),
    }


@app.get("/api/telemetry")
async def get_telemetry_summary():
    """
    Return pipeline telemetry and efficiency metrics across ingestion runs.
    Demonstrates multi-page batching, local table parsing, and heuristic pre-filtering savings.
    """
    from backend.telemetry import global_telemetry
    return global_telemetry.get_summary()


@app.get("/api/documents/{doc_id}/telemetry")
async def get_document_telemetry(doc_id: str):
    """Return telemetry for a specific document run if available."""
    from backend.telemetry import global_telemetry
    data = global_telemetry.get_run(doc_id)
    if not data:
        raise HTTPException(status_code=404, detail="No telemetry recorded for this document run.")
    return data


from pydantic import BaseModel


class CompareRequest(BaseModel):
    fact_id_1: str
    fact_id_2: str


@app.post("/api/relationships/compare")
async def compare_facts_on_demand(
    req: CompareRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    model: Optional[str] = Query(None),
):
    """
    Judge relationship between any two specific facts on-demand.
    """
    import os

    from backend.models import CandidatePair, Fact, MatchSource
    from backend.relation_judge import judge_single_pair

    from backend.config import DEFAULT_LLM_MODEL

    model = model or DEFAULT_LLM_MODEL
    effective_key = get_effective_key(x_api_key, model=model)
    if effective_key and effective_key.startswith("gsk_") and not model.startswith("groq/"):
        model = "groq/openai/gpt-oss-120b"

    if not effective_key:
        raise HTTPException(
            status_code=400,
            detail="API key required. Provide via X-API-Key header or set GROQ_API_KEY in .env.",
        )

    f1_dict = await db.get_fact(req.fact_id_1)
    f2_dict = await db.get_fact(req.fact_id_2)

    if not f1_dict or not f2_dict:
        raise HTTPException(status_code=404, detail="One or both facts not found")

    f1 = Fact(**f1_dict)
    f2 = Fact(**f2_dict)

    hint = "exact_scope" if f1.claim_fingerprint == f2.claim_fingerprint else "different_scope"
    pair = CandidatePair(
        fact_1=f1,
        fact_2=f2,
        match_source=MatchSource.STRUCTURAL,
        match_hint=hint,
        is_intra_document=(f1.source_doc_id == f2.source_doc_id),
    )

    relationship = await judge_single_pair(pair=pair, api_key=effective_key, model=model)
    if not await db.relationship_exists(f1.id, f2.id):
        await db.insert_relationship(relationship.model_dump())

    return {
        "relationship": relationship.model_dump(),
        "fact_1": f1.model_dump(),
        "fact_2": f2.model_dump(),
    }


# --- Normalization Registry Inspection ---


@app.get("/api/normalization/canonicals")
async def get_canonicals():
    """List canonical normalized entities and attributes in the registry."""
    subjects = await db.get_canonical_values("subject")
    attributes = await db.get_canonical_values("attribute")
    return {
        "subjects": [s["canonical_value"] for s in subjects],
        "attributes": [a["canonical_value"] for a in attributes],
    }


# --- Static Frontend ---
# Mount frontend directory to serve the UI at the root
_frontend_dir = PROJECT_ROOT / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
