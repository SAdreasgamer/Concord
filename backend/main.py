"""
Concord — Fact Knowledge Layer API.

FastAPI application entry point. Serves the REST API and static frontend.
"""

from __future__ import annotations

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
    """List all processed documents."""
    docs = await db.get_documents()
    return {"documents": docs}


@app.get("/api/documents/{doc_id}/facts")
async def get_document_facts(doc_id: str):
    """Get all facts from a specific document."""
    doc = await db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    facts = await db.get_facts(source_doc_id=doc_id)
    return {"document": doc, "facts": facts}


# --- Fact Endpoints ---


@app.get("/api/facts")
async def list_facts(
    source_doc_id: Optional[str] = Query(None),
    limit: int = Query(500, ge=1, le=2000),
):
    """List all facts, optionally filtered by document."""
    facts = await db.get_facts(source_doc_id=source_doc_id, limit=limit)
    return {"facts": facts, "count": len(facts)}


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


# --- Relationship Endpoints ---


@app.get("/api/relationships")
async def list_relationships(
    relation_type: Optional[str] = Query(None),
    limit: int = Query(500, ge=1, le=2000),
):
    """List all relationships, optionally filtered by type."""
    relationships = await db.get_relationships(
        relation_type=relation_type, limit=limit
    )

    # Enrich with fact data
    enriched = []
    for rel in relationships:
        fact_1 = await db.get_fact(rel["fact_id_1"])
        fact_2 = await db.get_fact(rel["fact_id_2"])
        enriched.append({**rel, "fact_1": fact_1, "fact_2": fact_2})

    return {"relationships": enriched, "count": len(enriched)}


# --- Key Validation ---


@app.post("/api/validate-key")
async def validate_key(
    x_api_key: str = Header(..., alias="X-API-Key"),
    model: Optional[str] = Query(None),
):
    """Validate an LLM API key with a lightweight test call."""
    import litellm

    from backend.config import DEFAULT_LLM_MODEL

    test_model = model or DEFAULT_LLM_MODEL

    try:
        response = litellm.completion(
            model=test_model,
            messages=[{"role": "user", "content": "Reply with exactly: ok"}],
            api_key=x_api_key,
            max_tokens=5,
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

    Parses the PDF into page-level text chunks, saves the document, and
    if an API key is provided (or configured in env), runs LLM fact extraction
    and canonical normalization.
    """
    import os
    import uuid

    from backend.config import UPLOAD_DIR
    from backend.fact_extractor import extract_document_facts
    from backend.pdf_parser import get_page_count, parse_pdf

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

    # Parse PDF into page chunks
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

    # Check for API key in header or environment
    effective_key = x_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    facts_extracted = []
    relationships_found = []

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
            )
            # Run candidate matching and relation judging across documents
            if facts_extracted:
                candidates = await matcher.find_candidates(new_facts=facts_extracted)
                if candidates:
                    from backend.relation_judge import judge_and_store_candidates

                    relationships_found = await judge_and_store_candidates(
                        candidates=candidates,
                        db=db,
                        api_key=effective_key,
                        model=model,
                    )
        except Exception as e:
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
            }

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
    """
    import os

    from backend.config import UPLOAD_DIR
    from backend.fact_extractor import extract_document_facts
    from backend.pdf_parser import parse_pdf
    from backend.relation_judge import judge_and_store_candidates

    doc = await db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = UPLOAD_DIR / f"{doc_id}.pdf"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Source PDF file not found on server")

    effective_key = x_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not effective_key:
        raise HTTPException(
            status_code=400,
            detail="API key required. Provide via X-API-Key header or set GEMINI_API_KEY.",
        )

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
    )

    relationships = []
    if facts:
        candidates = await matcher.find_candidates(new_facts=facts)
        if candidates:
            relationships = await judge_and_store_candidates(
                candidates=candidates,
                db=db,
                api_key=effective_key,
                model=model,
            )

    return {
        "doc_id": doc_id,
        "doc_name": doc["doc_name"],
        "fact_count": len(facts),
        "relationship_count": len(relationships),
        "facts": [f.model_dump() for f in facts],
        "relationships": [r.model_dump() for r in relationships],
    }


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

    effective_key = x_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not effective_key:
        raise HTTPException(
            status_code=400,
            detail="API key required. Provide via X-API-Key header or set GEMINI_API_KEY.",
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
