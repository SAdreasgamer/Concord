"""
LLM fact extraction module.

Sends page text to the LLM and parses structured, grounded facts.
No heuristics, no page classifiers, no local table extraction.
Just: page text → LLM → structured facts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Any, Optional

import litellm

from backend.config import DEFAULT_LLM_MODEL, OLLAMA_API_BASE
from backend.database import Database
from backend.models import ExtractedFact, Fact, PageChunk
from backend.prompts import (
    FACT_EXTRACTION_SYSTEM_PROMPT,
    FACT_EXTRACTION_USER_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)


# --- Utilities ---


def clean_snake_case(text: str) -> str:
    """Convert text to clean lowercase snake_case."""
    if not text:
        return "unspecified"
    cleaned = re.sub(r"[^\w\s-]", "", text.strip())
    cleaned = re.sub(r"[\s-]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_").lower()
    # Strip corporate suffixes
    for suffix in ("_limited", "_ltd", "_pvt_ltd", "_private_limited", "_inc", "_incorporated", "_corp", "_corporation", "_llc"):
        if cleaned.endswith(suffix) and len(cleaned) > len(suffix):
            cleaned = cleaned[: -len(suffix)].rstrip("_")
            break
    return cleaned or "unspecified"


def normalize_temporal_scope(scope: Optional[str]) -> str:
    """Normalize temporal scopes into canonical formats (e.g. fy2024, cy2023, q4_fy2024)."""
    if not scope:
        return "unspecified"
    s = scope.strip().lower()
    # Match quarters like Q4 FY24, Q4 2024
    m_quarter = re.search(r"(q[1-4])\s*(?:of\s*)?(?:fy\s*)?(\d{2,4})", s)
    if m_quarter:
        q = m_quarter.group(1)
        yr = int(m_quarter.group(2))
        if yr < 100:
            yr += 2000
        return f"{q}_fy{yr}"
    # Match fiscal year ranges like 2023-24, 2023/24, FY2023-24
    m_range = re.search(r"(?:fy\s*)?(\d{4})[-/](\d{2,4})", s)
    if m_range:
        start_yr = int(m_range.group(1))
        end_yr = int(m_range.group(2))
        if end_yr < 100:
            end_yr += (start_yr // 100) * 100
        return f"fy{end_yr}"
    # Match single FY like FY24, FY2024
    m_single = re.search(r"fy\s*(\d{2,4})", s)
    if m_single:
        yr = int(m_single.group(1))
        if yr < 100:
            yr += 2000
        return f"fy{yr}"
    # Match calendar years like 2023, 2024
    m_yr = re.search(r"\b(19\d\d|20\d\d)\b", s)
    if m_yr:
        return f"cy{m_yr.group(1)}"
    return clean_snake_case(s)


def normalize_attribute_name(attr: str) -> str:
    """Normalize metric attribute names to canonical forms."""
    c = clean_snake_case(attr)
    # Strip trailing noisy qualifiers
    for suffix in ("_rate", "_ratio", "_total", "_value", "_figure", "_level", "_estimate"):
        if c.endswith(suffix) and len(c) > len(suffix) + 3:
            c = c[: -len(suffix)]
            break
    # Domain-agnostic canonical mappings for common metric synonyms
    synonyms = {
        "profit_after_tax": "pat",
        "profit_after_tax_loss": "pat",
        "net_profit": "pat",
        "net_loss": "pat",
        "restated_profit": "pat",
        "restated_loss": "pat",
        "restated_loss_for_the_period": "pat",
        "restated_loss_for_the_year": "pat",
        "restated_profit_loss": "pat",
        "revenue_from_operations": "revenue",
        "total_revenue": "revenue",
        "total_income": "revenue",
        "real_gdp_growth": "gdp_growth",
        "gdp_growth": "gdp_growth",
        "growth_in_real_gdp": "gdp_growth",
        "headline_inflation": "cpi_inflation",
        "cpi_inflation": "cpi_inflation",
        "retail_inflation": "cpi_inflation",
        "gross_fiscal_deficit": "fiscal_deficit",
    }
    return synonyms.get(c, c)


def generate_claim_fingerprint(
    subject_normalized: str,
    attribute_normalized: str,
    temporal_scope: Optional[str] = None,
) -> str:
    """
    Generate a structural claim fingerprint.
    Format: `{subject}::{attribute}::{scope}`
    Facts with identical fingerprints are candidates for corroboration or contradiction.
    """
    subj = clean_snake_case(subject_normalized)
    attr = normalize_attribute_name(attribute_normalized)
    scope = normalize_temporal_scope(temporal_scope)
    return f"{subj}::{attr}::{scope}"


# --- JSON Parsing ---


def repair_and_parse_json(text: str) -> Optional[dict[str, Any]]:
    """Parse JSON from LLM response, handling markdown fences and truncation."""
    if not text or not text.strip():
        return None
    cleaned = text.strip()

    # Strip markdown fences
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()

    # Direct parse
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Repair truncated JSON
    last_brace = cleaned.rfind("}")
    if last_brace != -1:
        truncated = cleaned[: last_brace + 1].strip()
        for closer in ["]}", "}", "]"]:
            try:
                return json.loads(truncated + closer)
            except Exception:
                continue
        try:
            return json.loads(truncated)
        except Exception:
            pass

    # Regex fallback: find individual fact objects
    fact_objects = []
    for obj_match in re.finditer(r'\{[^{}]*"subject"[^{}]*\}', cleaned):
        try:
            obj = json.loads(obj_match.group(0))
            if isinstance(obj, dict) and "subject" in obj and "value" in obj:
                fact_objects.append(obj)
        except Exception:
            pass
    if fact_objects:
        return {"facts": fact_objects}

    return None


def parse_llm_facts(raw_text: str, default_page: int = 1) -> list[ExtractedFact]:
    """Parse LLM response into ExtractedFact objects."""
    data = repair_and_parse_json(raw_text)
    if not data:
        logger.warning("Could not parse JSON from LLM. Preview: %s", raw_text[:200])
        return []

    facts_raw = data.get("facts", []) if isinstance(data, dict) else data if isinstance(data, list) else []
    results: list[ExtractedFact] = []

    for item in facts_raw:
        if not isinstance(item, dict):
            continue

        subject = str(item.get("subject", "")).strip()
        attribute = str(item.get("attribute", "")).strip()
        value = str(item.get("value", "")).strip()
        evidence_quote = str(item.get("evidence_quote", "")).strip()

        if not subject or not attribute or not value:
            continue
        # Skip overly long values (prose dumps, not atomic facts)
        if len(value) > 60 or len(value.split()) > 8:
            continue
        if len(attribute) > 60 or len(attribute.split()) > 8:
            continue

        if not evidence_quote:
            evidence_quote = f"{subject} {attribute}: {value}"

        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.85))))
        except (ValueError, TypeError):
            confidence = 0.85

        results.append(ExtractedFact(
            subject=subject,
            subject_normalized=str(item.get("subject_normalized") or subject),
            attribute=attribute,
            attribute_normalized=str(item.get("attribute_normalized") or attribute),
            value=value,
            unit=item.get("unit"),
            temporal_scope=item.get("temporal_scope"),
            conditions=item.get("conditions"),
            evidence_quote=evidence_quote,
            page=default_page,
            confidence=confidence,
            extraction_group_id=item.get("extraction_group_id"),
        ))

    return results


# --- LLM Extraction ---


async def extract_facts_from_page(
    chunk: PageChunk,
    api_key: str,
    model: Optional[str] = None,
) -> list[ExtractedFact]:
    """Send a single page to the LLM and extract facts."""
    target_model = model or DEFAULT_LLM_MODEL

    page_text = chunk.text[:2400] if len(chunk.text) > 2400 else chunk.text
    user_prompt = FACT_EXTRACTION_USER_PROMPT_TEMPLATE.format(
        doc_name=chunk.doc_name,
        page_number=chunk.page_number,
        page_text=page_text,
    )

    max_retries = 4
    for attempt in range(max_retries):
        try:
            extra_kwargs = {}
            if target_model.startswith("ollama/"):
                extra_kwargs["api_base"] = OLLAMA_API_BASE

            response = await litellm.acompletion(
                model=target_model,
                messages=[
                    {"role": "system", "content": FACT_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                api_key=api_key if not target_model.startswith("ollama/") else None,
                temperature=0.1,
                max_tokens=350,
                timeout=25,
                **extra_kwargs,
            )
            raw_text = response.choices[0].message.content or ""
            return parse_llm_facts(raw_text, default_page=chunk.page_number)

        except Exception as e:
            err_msg = str(e).lower()
            is_rate_limit = "429" in err_msg or "quota" in err_msg or "rate limit" in err_msg or "resource_exhausted" in err_msg
            if is_rate_limit and attempt < max_retries - 1:
                delay = 3.0 * (attempt + 1)
                m = re.search(r"try again in ([\d\.]+)s", err_msg)
                if m:
                    delay = max(delay, float(m.group(1)) + 1.0)
                logger.warning("Rate limit on page %d (attempt %d/%d). Waiting %.1fs...", chunk.page_number, attempt + 1, max_retries, delay)
                await asyncio.sleep(delay)
            else:
                logger.error("Extraction failed for page %d: %s", chunk.page_number, e)
                return []
    return []


# --- Pipeline ---


async def extract_document_facts(
    chunks: list[PageChunk],
    doc_id: str,
    doc_name: str,
    db: Database,
    api_key: str,
    model: Optional[str] = None,
    max_pages: Optional[int] = None,
    high_signal_only: bool = True,
) -> list[Fact]:
    """
    Full extraction pipeline:
    1. Select high-signal data pages (or user specified range)
    2. Extract grounded facts from each selected page
    3. Generate normalized claim fingerprints
    4. Store in database
    """
    from backend.pdf_parser import get_high_signal_chunks

    if max_pages:
        # Respect the user's page limit gate (e.g. first 15 pages)
        subset = chunks[:max_pages]
        non_fm = [c for c in subset if not c.is_front_matter and len(c.text.strip()) >= 50]
        valid_subset = non_fm if non_fm else subset
        if high_signal_only and len(valid_subset) > 3:
            chunks_to_process = get_high_signal_chunks(valid_subset, max_chunks=3)
        else:
            chunks_to_process = valid_subset
        logger.info(
            "Selected %d high-signal pages within first %d-page gate for '%s': %s",
            len(chunks_to_process),
            max_pages,
            doc_name,
            [c.page_number for c in chunks_to_process],
        )
    elif high_signal_only and len(chunks) > 3:
        chunks_to_process = get_high_signal_chunks(chunks, max_chunks=3)
        logger.info(
            "Selected %d high-signal data pages out of %d total pages for '%s': %s",
            len(chunks_to_process),
            len(chunks),
            doc_name,
            [c.page_number for c in chunks_to_process],
        )
    else:
        chunks_to_process = chunks

    all_extracted: list[ExtractedFact] = []

    for chunk in chunks_to_process:
        # Skip pages with very little text (covers, blank pages)
        if len(chunk.text.strip()) < 50:
            logger.debug("Skipping page %d: too little text (%d chars)", chunk.page_number, len(chunk.text))
            continue

        page_facts = await extract_facts_from_page(chunk, api_key, model)
        all_extracted.extend(page_facts)
        logger.info("Page %d: extracted %d facts", chunk.page_number, len(page_facts))

        target_model = model or DEFAULT_LLM_MODEL
        # Polite pacing between pages for rate limits
        if "groq" in target_model.lower():
            await asyncio.sleep(2.0)

    if not all_extracted:
        return []

    # Fingerprint and store
    saved_facts: list[Fact] = []
    for item in all_extracted:
        subject_norm = clean_snake_case(item.subject_normalized)
        attr_norm = clean_snake_case(item.attribute_normalized)
        fingerprint = generate_claim_fingerprint(subject_norm, attr_norm, item.temporal_scope)

        group_id = f"{doc_id}_{item.page}_{item.extraction_group_id}" if item.extraction_group_id else None

        fact_dict = {
            "id": str(uuid.uuid4()),
            "subject": item.subject,
            "subject_normalized": subject_norm,
            "attribute": item.attribute,
            "attribute_normalized": attr_norm,
            "value": item.value,
            "unit": item.unit,
            "temporal_scope": item.temporal_scope,
            "conditions": item.conditions,
            "claim_fingerprint": fingerprint,
            "extraction_group_id": group_id,
            "source_doc": doc_name,
            "source_doc_id": doc_id,
            "page": item.page,
            "evidence_quote": item.evidence_quote,
            "confidence": item.confidence,
        }

        fact_id = await db.insert_fact(fact_dict)
        fact_dict["id"] = fact_id
        saved_facts.append(Fact(**fact_dict))

    logger.info("Stored %d facts for '%s'", len(saved_facts), doc_name)
    return saved_facts
