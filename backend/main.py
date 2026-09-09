"""
Concord — Fact Knowledge Layer API.

FastAPI application. Serves the REST API and static frontend.
Clean, minimal endpoints — no over-engineering.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import litellm
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import PROJECT_ROOT
from backend.database import Database

logger = logging.getLogger(__name__)

# --- Singletons ---
db = Database()


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect()
    yield
    await db.close()


# --- App ---
app = FastAPI(
    title="Concord",
    description="Fact Knowledge Layer — extract, ground, and compare facts across PDF documents.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Helpers ---


def resolve_model_and_key(
    model: Optional[str], x_api_key: Optional[str]
) -> tuple[str, Optional[str]]:
    """Resolve the LLM model and API key from request headers or environment."""
    from backend.config import (
        DEFAULT_LLM_MODEL, GROQ_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY,
    )

    effective_model = model or DEFAULT_LLM_MODEL
    effective_key = x_api_key

    if not effective_key:
        if "groq" in effective_model.lower() and GROQ_API_KEY:
            effective_key = GROQ_API_KEY
        elif "gemini" in effective_model.lower() and GEMINI_API_KEY:
            effective_key = GEMINI_API_KEY
        elif "gpt" in effective_model.lower() and OPENAI_API_KEY:
            effective_key = OPENAI_API_KEY
        elif "claude" in effective_model.lower() and ANTHROPIC_API_KEY:
            effective_key = ANTHROPIC_API_KEY
        elif GROQ_API_KEY:
            effective_key = GROQ_API_KEY
        elif GEMINI_API_KEY:
            effective_key = GEMINI_API_KEY

    return effective_model, effective_key


# --- Health ---


@app.get("/health")
async def health():
    stats = await db.get_stats()
    return {"status": "ok", "stats": stats}


@app.post("/api/validate-key")
async def validate_key(
    model: Optional[str] = Query(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    """Validate LLM model connectivity and credentials."""
    effective_model, effective_key = resolve_model_and_key(model, x_api_key)
    if not effective_key and not effective_model.startswith("ollama/"):
        return {"valid": False, "model": effective_model, "message": "No API key found in request or .env."}

    try:
        extra_kwargs = {}
        if effective_model.startswith("ollama/"):
            from backend.config import OLLAMA_API_BASE
            extra_kwargs["api_base"] = OLLAMA_API_BASE

        await litellm.acompletion(
            model=effective_model,
            messages=[{"role": "user", "content": "ping"}],
            api_key=effective_key if not effective_model.startswith("ollama/") else None,
            max_tokens=10,
            timeout=15,
            **extra_kwargs,
        )
        return {"valid": True, "model": effective_model, "message": f"Connected to {effective_model} successfully!"}
    except Exception as e:
        logger.warning("Key validation failed for %s: %s", effective_model, e)
        return {"valid": False, "model": effective_model, "message": str(e)}


# --- Reset ---


@app.post("/api/reset")
async def reset():
    """Wipe all data and uploaded files."""
    await db.clear_all_data()
    from backend.config import UPLOAD_DIR
    deleted = 0
    if UPLOAD_DIR.exists():
        for f in UPLOAD_DIR.glob("*.pdf"):
            try:
                f.unlink()
                deleted += 1
            except Exception:
                pass
    return {"status": "success", "message": "All data wiped", "deleted_files": deleted}


# --- Documents ---


@app.get("/api/documents")
async def list_documents():
    return {"documents": await db.get_documents()}


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    doc = await db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    facts = await db.get_facts(source_doc_id=doc_id)
    rels = await db.get_relationships_for_document(doc_id)
    return {"document": {**doc, "fact_count": len(facts), "relationship_count": len(rels)}}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    if not await db.delete_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted", "doc_id": doc_id}


# --- Facts ---


@app.get("/api/facts")
async def list_facts(
    source_doc_id: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(500, ge=1, le=2000),
):
    facts = await db.get_facts(source_doc_id=source_doc_id, limit=limit)
    if search:
        q = search.strip().lower()
        facts = [
            f for f in facts
            if q in f["subject"].lower()
            or q in f["attribute"].lower()
            or q in f["value"].lower()
            or q in f["evidence_quote"].lower()
        ]
    return {"facts": facts, "count": len(facts)}


@app.get("/api/facts/{fact_id}")
async def get_fact_detail(fact_id: str):
    fact = await db.get_fact(fact_id)
    if not fact:
        raise HTTPException(status_code=404, detail="Fact not found")
    relationships = await db.get_relationships_for_fact(fact_id)
    # Enrich with related facts
    related_facts = []
    for rel in relationships:
        other_id = rel["fact_id_2"] if rel["fact_id_1"] == fact_id else rel["fact_id_1"]
        other = await db.get_fact(other_id)
        if other:
            related_facts.append(other)
    return {"fact": fact, "relationships": relationships, "related_facts": related_facts}


@app.get("/api/facts/{fact_id}/related")
async def get_fact_related(fact_id: str):
    """Return relationships involving this fact, enriched with target_fact."""
    relationships = await db.get_relationships_for_fact(fact_id)
    enriched = []
    for rel in relationships:
        other_id = rel["fact_id_2"] if rel["fact_id_1"] == fact_id else rel["fact_id_1"]
        other = await db.get_fact(other_id)
        enriched.append({**rel, "target_fact": other})
    return {"relationships": enriched}


# --- Relationships ---


class CompareRequest(BaseModel):
    fact_id_1: str
    fact_id_2: str


@app.post("/api/relationships/compare")
async def compare_facts(
    req: CompareRequest,
    model: Optional[str] = Query(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    """Ad-hoc judge between any two arbitrary facts."""
    from backend.models import CandidatePair, Fact
    from backend.relation_judge import judge_single_pair

    f1_dict = await db.get_fact(req.fact_id_1)
    f2_dict = await db.get_fact(req.fact_id_2)
    if not f1_dict or not f2_dict:
        raise HTTPException(status_code=404, detail="One or both facts not found")

    f1 = Fact(**f1_dict)
    f2 = Fact(**f2_dict)
    eff_model, eff_key = resolve_model_and_key(model, x_api_key)

    candidate = CandidatePair(
        fact_1=f1,
        fact_2=f2,
        match_source=MatchSource.FUZZY,
        is_intra_document=(f1.source_doc_id == f2.source_doc_id),
    )
    rel = await judge_single_pair(candidate, api_key=eff_key or "", model=eff_model)
    if rel:
        await db.insert_relationship(rel)
        return {"relationship": rel.model_dump()}
    raise HTTPException(status_code=500, detail="Relation judge could not evaluate pair.")


@app.get("/api/relationships")
async def list_relationships(
    relation_type: Optional[str] = Query(None),
    limit: int = Query(500, ge=1, le=2000),
):
    rels = await db.get_relationships(relation_type=relation_type, limit=limit)
    # Enrich each relationship with its two facts
    enriched = []
    for rel in rels:
        f1 = await db.get_fact(rel["fact_id_1"])
        f2 = await db.get_fact(rel["fact_id_2"])
        enriched.append({**rel, "fact_1": f1, "fact_2": f2})
    return {"relationships": enriched, "count": len(enriched)}


@app.get("/api/relationships/{rel_id}")
async def get_relationship(rel_id: str):
    rel = await db.get_relationship(rel_id)
    if not rel:
        raise HTTPException(status_code=404, detail="Relationship not found")
    f1 = await db.get_fact(rel["fact_id_1"])
    f2 = await db.get_fact(rel["fact_id_2"])
    return {"relationship": rel, "fact_1": f1, "fact_2": f2}


@app.get("/api/documents/{doc_id}/relationships")
async def get_document_relationships(doc_id: str):
    doc = await db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    rels = await db.get_relationships_for_document(doc_id)
    enriched = []
    for rel in rels:
        f1 = await db.get_fact(rel["fact_id_1"])
        f2 = await db.get_fact(rel["fact_id_2"])
        enriched.append({**rel, "fact_1": f1, "fact_2": f2})
    return {"document": doc, "relationships": enriched, "count": len(enriched)}


# --- Normalization & Telemetry Helpers ---


@app.get("/api/normalization/canonicals")
async def get_canonicals():
    """Return unique subjects and attributes derived from extracted facts."""
    facts = await db.get_facts(limit=2000)
    subjects = sorted(list(set(f["subject"] for f in facts if f.get("subject"))))
    attributes = sorted(list(set(f["attribute"] for f in facts if f.get("attribute"))))
    return {"subjects": subjects, "attributes": attributes}


@app.get("/api/export")
async def export_data():
    """Export complete knowledge layer snapshot as JSON."""
    docs = await db.get_documents()
    facts = await db.get_facts(limit=5000)
    rels = await db.get_relationships(limit=5000)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "counts": {"documents": len(docs), "facts": len(facts), "relationships": len(rels)},
        "documents": docs,
        "facts": facts,
        "relationships": rels,
    }


@app.get("/api/telemetry")
async def get_telemetry():
    """Return telemetry and cost reduction metrics for the frontend."""
    stats = await db.get_stats()
    return {
        "cumulative_local_facts": 0,
        "local_extraction_ratio_pct": 0,
        "cumulative_skipped_pages": 0,
        "pages_saved_pct": 0,
        "recent_runs": [],
        "naive_baseline": {
            "calls_saved": 0,
            "reduction_factor": "1.0x",
            "cost_saved_usd": 0.0,
        },
    }


# --- Sample Datasets ---


def _discover_sample_datasets() -> dict:
    """Dynamically discover sample datasets from starter-datasets/ directory."""
    datasets = {}
    starter_dir = PROJECT_ROOT / "starter-datasets"
    if not starter_dir.exists():
        return datasets

    for category_dir in sorted(starter_dir.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith("."):
            continue
        for pdf_file in sorted(category_dir.glob("*.pdf")):
            key = pdf_file.stem.replace("-", "_")
            # Generate a human-readable title from filename
            title = pdf_file.stem.replace("-", " ").title()
            # Remove leading numbers like "01 "
            title = title.lstrip("0123456789 ")
            datasets[key] = {
                "title": title,
                "category": category_dir.name.replace("-", " ").title(),
                "filename": pdf_file.name,
                "rel_path": str(pdf_file.relative_to(PROJECT_ROOT)),
            }
    return datasets


@app.get("/api/sample-datasets")
async def list_sample_datasets():
    datasets = _discover_sample_datasets()
    return {
        "samples": [
            {"key": k, "title": v["title"], "category": v["category"], "filename": v["filename"]}
            for k, v in datasets.items()
        ]
    }


@app.post("/api/sample-datasets/{dataset_key}/load")
async def load_sample_dataset(
    dataset_key: str,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    model: Optional[str] = Query(None),
    max_pages: Optional[int] = Query(None),
):
    """Load a bundled starter PDF and process it."""
    from backend.config import DEFAULT_PAGE_LIMIT, UPLOAD_DIR
    from backend.fact_extractor import extract_document_facts
    from backend.matcher import find_candidates
    from backend.pdf_parser import get_page_count, parse_pdf
    from backend.relation_judge import judge_and_store_candidates

    effective_max_pages = DEFAULT_PAGE_LIMIT if max_pages is None else (None if max_pages <= 0 else max_pages)

    datasets = _discover_sample_datasets()
    sample = datasets.get(dataset_key)
    if not sample:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_key}' not found. Available: {list(datasets.keys())}")

    file_path = PROJECT_ROOT / sample["rel_path"]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Sample PDF file missing")

    file_bytes = file_path.read_bytes()
    doc_id = str(uuid.uuid4())
    doc_name = sample["filename"]

    chunks = parse_pdf(file_bytes, doc_name=doc_name, doc_id=doc_id, max_pages=effective_max_pages)
    page_count = len(chunks)

    save_path = UPLOAD_DIR / f"{doc_id}.pdf"
    save_path.write_bytes(file_bytes)
    await db.insert_document(doc_id, doc_name, page_count)

    model, effective_key = resolve_model_and_key(model, x_api_key)

    facts = []
    relationships = []

    if effective_key:
        try:
            facts = await extract_document_facts(
                chunks=chunks, doc_id=doc_id, doc_name=doc_name,
                db=db, api_key=effective_key, model=model, max_pages=effective_max_pages,
            )
            if facts:
                candidates = await find_candidates(new_facts=facts, db=db)
                if candidates:
                    relationships = await judge_and_store_candidates(
                        candidates=candidates, db=db, api_key=effective_key, model=model,
                    )
        except Exception as e:
            logger.error("Processing error: %s", e)
            return {
                "doc_id": doc_id, "doc_name": doc_name, "page_count": page_count,
                "fact_count": len(facts), "relationship_count": len(relationships),
                "status": "partial_error", "message": str(e),
            }

    return {
        "doc_id": doc_id,
        "doc_name": doc_name,
        "page_count": page_count,
        "fact_count": len(facts),
        "relationship_count": len(relationships),
        "status": "extracted" if effective_key else "parsed_only",
        "message": f"Extracted {len(facts)} facts, discovered {len(relationships)} relationships." if effective_key else "Parsed. Provide API key to extract.",
        "facts_preview": [f.model_dump() for f in facts[:15]],
        "relationships_preview": [r.model_dump() for r in relationships[:10]],
    }


# --- PDF Upload ---


@app.post("/api/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    model: Optional[str] = Query(None),
    max_pages: Optional[int] = Query(None),
):
    """
    Upload a PDF, extract facts, match across documents, judge relationships.
    Pipeline: Parse → Extract (LLM) → Fingerprint → Match → Judge (LLM) → Store
    """
    from backend.config import DEFAULT_PAGE_LIMIT, UPLOAD_DIR
    from backend.fact_extractor import extract_document_facts
    from backend.matcher import find_candidates
    from backend.pdf_parser import get_page_count, parse_pdf
    from backend.relation_judge import judge_and_store_candidates

    effective_max_pages = DEFAULT_PAGE_LIMIT if max_pages is None else (None if max_pages <= 0 else max_pages)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    doc_id = str(uuid.uuid4())
    doc_name = file.filename

    try:
        chunks = parse_pdf(file_bytes, doc_name=doc_name, doc_id=doc_id, max_pages=effective_max_pages)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text found in PDF.")

    save_path = UPLOAD_DIR / f"{doc_id}.pdf"
    save_path.write_bytes(file_bytes)

    page_count = len(chunks)
    await db.insert_document(doc_id, doc_name, page_count)

    model, effective_key = resolve_model_and_key(model, x_api_key)

    facts = []
    relationships = []

    if effective_key:
        try:
            facts = await extract_document_facts(
                chunks=chunks, doc_id=doc_id, doc_name=doc_name,
                db=db, api_key=effective_key, model=model, max_pages=effective_max_pages,
            )
            if facts:
                candidates = await find_candidates(new_facts=facts, db=db)
                if candidates:
                    relationships = await judge_and_store_candidates(
                        candidates=candidates, db=db, api_key=effective_key, model=model,
                    )
        except Exception as e:
            logger.error("Processing error: %s", e)
            return {
                "doc_id": doc_id, "doc_name": doc_name, "page_count": page_count,
                "fact_count": len(facts), "relationship_count": len(relationships),
                "status": "partial_error", "message": str(e),
                "facts_preview": [f.model_dump() for f in facts[:15]],
                "relationships_preview": [r.model_dump() for r in relationships[:10]],
            }

    return {
        "doc_id": doc_id,
        "doc_name": doc_name,
        "page_count": page_count,
        "fact_count": len(facts),
        "relationship_count": len(relationships),
        "status": "extracted" if effective_key else "parsed_only",
        "message": (
            f"Extracted {len(facts)} facts and discovered {len(relationships)} relationships."
            if effective_key
            else "Parsed. Pass X-API-Key header or set GROQ_API_KEY in .env to extract facts."
        ),
        "facts_preview": [f.model_dump() for f in facts[:15]],
        "relationships_preview": [r.model_dump() for r in relationships[:10]],
    }


@app.post("/api/documents/{doc_id}/process")
async def process_document(
    doc_id: str,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    model: Optional[str] = Query(None),
    max_pages: Optional[int] = Query(None),
):
    """Re-process or initially process a document that was saved without an API key."""
    from backend.config import DEFAULT_PAGE_LIMIT, UPLOAD_DIR
    from backend.fact_extractor import extract_document_facts
    from backend.matcher import find_candidates
    from backend.pdf_parser import parse_pdf
    from backend.relation_judge import judge_and_store_candidates

    effective_max_pages = DEFAULT_PAGE_LIMIT if max_pages is None else (None if max_pages <= 0 else max_pages)

    doc = await db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    pdf_path = UPLOAD_DIR / f"{doc_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Stored PDF file not found on disk")

    doc_name = doc.get("doc_name") or doc.get("filename") or "document.pdf"
    file_bytes = pdf_path.read_bytes()
    chunks = parse_pdf(file_bytes, doc_name=doc_name, doc_id=doc_id, max_pages=effective_max_pages)
    eff_model, eff_key = resolve_model_and_key(model, x_api_key)
    if not eff_key and not eff_model.startswith("ollama/"):
        raise HTTPException(status_code=400, detail="No API key found in request or .env")

    facts = await extract_document_facts(
        chunks=chunks, doc_id=doc_id, doc_name=doc_name,
        db=db, api_key=eff_key or "", model=eff_model, max_pages=effective_max_pages,
    )
    relationships = []
    if facts:
        candidates = await find_candidates(new_facts=facts, db=db)
        if candidates:
            relationships = await judge_and_store_candidates(
                candidates=candidates, db=db, api_key=eff_key or "", model=eff_model,
            )

    return {
        "doc_id": doc_id,
        "doc_name": doc["filename"],
        "fact_count": len(facts),
        "relationship_count": len(relationships),
        "status": "extracted",
    }


# --- Static Frontend ---
_frontend_dir = PROJECT_ROOT / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
