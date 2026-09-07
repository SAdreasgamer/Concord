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

# --- Database singleton ---
db = Database()


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    await db.connect()
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


# --- Static Frontend ---
# Mount frontend directory to serve the UI at the root
_frontend_dir = PROJECT_ROOT / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
