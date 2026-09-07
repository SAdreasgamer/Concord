"""
LLM fact extraction module.

Sends page chunks to the LLM and parses structured, grounded facts.
Decomposes compound statements, extracts verbatim evidence quotes,
and normalizes entities/attributes via NormalizationRegistry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Optional

import litellm

from backend.config import DEFAULT_LLM_MODEL
from backend.database import Database
from backend.models import ExtractedFact, Fact, PageChunk
from backend.normalization import NormalizationRegistry, generate_claim_fingerprint
from backend.prompts import (
    FACT_EXTRACTION_SYSTEM_PROMPT,
    FACT_EXTRACTION_USER_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)


def parse_llm_facts(raw_text: str, default_page: int = 1) -> list[ExtractedFact]:
    """
    Parse LLM response text into a list of ExtractedFact objects.

    Extracts JSON even if wrapped in markdown fences (```json ... ```)
    or preceded by narrative text.
    """
    if not raw_text or not raw_text.strip():
        return []

    # Strip markdown code blocks
    cleaned = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse LLM JSON: %s. Raw preview: %s", e, cleaned[:200])
        # Attempt minimal regex extraction if JSON object is malformed
        json_match = re.search(r"\{[\s\S]*\}", cleaned)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
            except Exception:
                return []
        else:
            return []

    # The prompt requests {"facts": [...]}
    facts_raw = []
    if isinstance(data, dict):
        facts_raw = data.get("facts", [])
    elif isinstance(data, list):
        facts_raw = data

    results: list[ExtractedFact] = []
    for item in facts_raw:
        if not isinstance(item, dict):
            continue

        # Validate minimum required fields
        subject = str(item.get("subject", "")).strip()
        attribute = str(item.get("attribute", "")).strip()
        value = str(item.get("value", "")).strip()
        evidence_quote = str(item.get("evidence_quote", "")).strip()

        if not subject or not attribute or not value:
            continue

        # If evidence quote is missing, fallback to attribute + value rather than discarding
        if not evidence_quote:
            evidence_quote = f"{subject} {attribute}: {value}"

        try:
            confidence = float(item.get("confidence", 0.85))
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.85

        try:
            page = int(item.get("page", default_page))
        except (ValueError, TypeError):
            page = default_page

        extracted = ExtractedFact(
            subject=subject,
            subject_normalized=str(item.get("subject_normalized") or subject),
            attribute=attribute,
            attribute_normalized=str(item.get("attribute_normalized") or attribute),
            value=value,
            unit=item.get("unit"),
            temporal_scope=item.get("temporal_scope"),
            conditions=item.get("conditions"),
            evidence_quote=evidence_quote,
            page=page,
            confidence=confidence,
            extraction_group_id=item.get("extraction_group_id"),
        )
        results.append(extracted)

    return results


async def extract_facts_from_chunk(
    chunk: PageChunk,
    api_key: str,
    model: Optional[str] = None,
) -> list[ExtractedFact]:
    """
    Call the LLM to extract facts from a single PageChunk.

    Args:
        chunk: The PageChunk containing document text and page metadata.
        api_key: User's LLM API key.
        model: Model identifier (defaults to DEFAULT_LLM_MODEL).

    Returns:
        List of raw ExtractedFact models.
    """
    target_model = model or DEFAULT_LLM_MODEL

    user_prompt = FACT_EXTRACTION_USER_PROMPT_TEMPLATE.format(
        doc_name=chunk.doc_name,
        page_number=chunk.page_number,
        page_text=chunk.text,
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # LiteLLM handles Google, OpenAI, Anthropic through a unified interface
            response = await litellm.acompletion(
                model=target_model,
                messages=[
                    {"role": "system", "content": FACT_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                api_key=api_key,
                temperature=0.1,
                max_tokens=4000,
                response_format={"type": "json_object"},
            )
            raw_text = response.choices[0].message.content or ""
            return parse_llm_facts(raw_text, default_page=chunk.page_number)
        except Exception as e:
            err_msg = str(e).lower()
            is_rate_limit = (
                isinstance(e, litellm.RateLimitError)
                or "429" in err_msg
                or "quota" in err_msg
                or "resource_exhausted" in err_msg
            )
            if is_rate_limit and attempt < max_retries - 1:
                delay = 5.0 * (attempt + 1)
                logger.warning(
                    "Rate limit encountered on '%s' page %d (attempt %d/%d). Backing off for %.1fs...",
                    chunk.doc_name,
                    chunk.page_number,
                    attempt + 1,
                    max_retries,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "LLM extraction error for '%s' page %d: %s",
                    chunk.doc_name,
                    chunk.page_number,
                    e,
                )
                raise


async def process_and_store_facts(
    extracted_facts: list[ExtractedFact],
    doc_id: str,
    doc_name: str,
    db: Database,
    registry: NormalizationRegistry,
) -> list[Fact]:
    """
    Resolve extracted facts against the NormalizationRegistry,
    generate structural fingerprints, and persist to the database.

    Args:
        extracted_facts: Raw facts from LLM parsing.
        doc_id: ID of the source document.
        doc_name: Name of the source document.
        db: Database instance.
        registry: NormalizationRegistry for resolving entity/attribute keys.

    Returns:
        List of persisted Fact models.
    """
    saved_facts: list[Fact] = []

    for item in extracted_facts:
        # 1. Resolve normalized subject against canonical registry
        subject_norm = await registry.resolve(
            key_type="subject",
            raw_value=item.subject,
            proposed_normalized=item.subject_normalized,
        )

        # 2. Resolve normalized attribute against canonical registry
        attr_norm = await registry.resolve(
            key_type="attribute",
            raw_value=item.attribute,
            proposed_normalized=item.attribute_normalized,
        )

        # 3. Generate deterministic claim fingerprint
        fingerprint = generate_claim_fingerprint(
            subject_normalized=subject_norm,
            attribute_normalized=attr_norm,
            temporal_scope=item.temporal_scope,
        )

        # 4. Generate unique group ID if decomposition group is present
        group_id = None
        if item.extraction_group_id:
            group_id = f"{doc_id}_{item.page}_{item.extraction_group_id}"

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

        # Store in database
        fact_id = await db.insert_fact(fact_dict)
        fact_dict["id"] = fact_id
        saved_facts.append(Fact(**fact_dict))

    logger.info(
        "Persisted %d facts for document '%s' (%s)",
        len(saved_facts),
        doc_name,
        doc_id,
    )
    return saved_facts


async def extract_facts_from_batch(
    chunks: list[PageChunk],
    api_key: str,
    model: Optional[str] = None,
) -> list[ExtractedFact]:
    """
    Extract facts from MULTIPLE pages in a SINGLE LLM call.

    Batches 4-6 pages into one prompt with clear page delimiters.
    This is the key cost optimization: 27 pages → ~5 API calls instead of 27.
    """
    if not chunks:
        return []

    # If only one chunk, fall back to single extraction
    if len(chunks) == 1:
        return await extract_facts_from_chunk(chunks[0], api_key, model)

    target_model = model or DEFAULT_LLM_MODEL

    # Build multi-page prompt with clear delimiters
    page_sections = []
    for chunk in chunks:
        page_sections.append(
            f"--- PAGE {chunk.page_number} (from: {chunk.doc_name}) ---\n"
            f"{chunk.text}\n"
            f"--- END PAGE {chunk.page_number} ---"
        )

    combined_text = "\n\n".join(page_sections)
    page_range = f"{chunks[0].page_number}-{chunks[-1].page_number}"

    user_prompt = (
        f"Document: {chunks[0].doc_name}\n"
        f"Pages: {page_range} ({len(chunks)} pages)\n\n"
        f"{combined_text}\n\n"
        f"Extract all atomic, verifiable facts from ALL {len(chunks)} pages above. "
        f"Ensure every fact includes the correct page number and an exact verbatim "
        f"evidence_quote from that specific page's text."
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await litellm.acompletion(
                model=target_model,
                messages=[
                    {"role": "system", "content": FACT_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                api_key=api_key,
                temperature=0.1,
                max_tokens=8000,  # More tokens for multi-page response
                response_format={"type": "json_object"},
            )
            raw_text = response.choices[0].message.content or ""

            # Track token usage for telemetry
            input_tokens = getattr(response.usage, "prompt_tokens", 0) if response.usage else 0
            output_tokens = getattr(response.usage, "completion_tokens", 0) if response.usage else 0

            facts = parse_llm_facts(raw_text, default_page=chunks[0].page_number)

            logger.info(
                "Batch extraction: %d facts from pages %s (%d input tokens, %d output tokens)",
                len(facts),
                page_range,
                input_tokens,
                output_tokens,
            )
            return facts

        except Exception as e:
            err_msg = str(e).lower()
            is_rate_limit = (
                isinstance(e, litellm.RateLimitError)
                or "429" in err_msg
                or "quota" in err_msg
                or "resource_exhausted" in err_msg
            )
            if is_rate_limit:
                if attempt < max_retries - 1 and "quota" not in err_msg:
                    delay = 3.0 * (attempt + 1)
                    logger.warning(
                        "Rate limit on batch pages %s (attempt %d/%d). Backing off %.1fs...",
                        page_range,
                        attempt + 1,
                        max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.warning("Batch extraction hit rate limit / quota on pages %s. Skipping batch without per-page fallback.", page_range)
                    raise e
            else:
                logger.error("Batch extraction failed for pages %s: %s", page_range, e)
                # Fallback: try individual pages ONLY for format/JSON errors
                logger.info("Falling back to individual page extraction for pages %s", page_range)
                all_facts = []
                for chunk in chunks:
                    try:
                        facts = await extract_facts_from_chunk(chunk, api_key, model)
                        all_facts.extend(facts)
                    except Exception as inner_e:
                        logger.warning("Skipping page %d: %s", chunk.page_number, inner_e)
                return all_facts


async def extract_document_facts(
    chunks: list[PageChunk],
    doc_id: str,
    doc_name: str,
    db: Database,
    registry: NormalizationRegistry,
    api_key: str,
    model: Optional[str] = None,
    max_pages: Optional[int] = None,
    telemetry: Optional["PipelineTelemetry"] = None,
) -> list[Fact]:
    """
    HYBRID extraction pipeline — the core architectural differentiator.

    Instead of blindly sending every page to the LLM like ChatGPT would,
    Concord uses a 3-stage approach:

    1. CLASSIFY: Heuristic page classifier skips junk pages (TOC, covers,
       disclaimers) without touching the LLM.
    2. LOCAL EXTRACT: Pages with structured tables get facts extracted
       locally using PyMuPDF's table parser — zero API calls.
    3. BATCH LLM: Remaining pages are batched into groups of 6 for
       a single LLM call — 5x fewer API calls.

    Result: A 27-page PDF that would cost 27 API calls now costs ~4-5.
    """
    from backend.page_classifier import classify_page, PageType
    from backend.table_extractor import extract_facts_from_table
    from backend.telemetry import PipelineTelemetry

    if telemetry is None:
        telemetry = PipelineTelemetry()

    all_extracted: list[ExtractedFact] = []
    chunks_to_process = chunks[:max_pages] if max_pages else chunks
    telemetry.total_pages = len(chunks_to_process)

    # --- Stage 1: CLASSIFY pages ---
    stage_classify = telemetry.start_stage("page_classification")
    pages_for_llm: list[PageChunk] = []

    for chunk in chunks_to_process:
        classification = classify_page(
            text=chunk.text,
            page_number=chunk.page_number,
            has_tables=bool(chunk.tables),
            table_row_count=chunk.table_row_count,
        )

        if not classification.should_send_to_llm:
            telemetry.record_page_skip(classification.page_type.value)
            logger.debug(
                "Skipping page %d: %s (%s)",
                chunk.page_number,
                classification.page_type.value,
                classification.reason,
            )
            continue

        # --- Stage 2: LOCAL TABLE EXTRACTION (no LLM) ---
        if classification.should_extract_tables_locally and chunk.tables:
            for table_data in chunk.tables:
                local_facts = extract_facts_from_table(
                    table_data=table_data,
                    doc_name=doc_name,
                    page_number=chunk.page_number,
                )
                if local_facts:
                    all_extracted.extend(local_facts)
                    telemetry.facts_from_local_tables += len(local_facts)

        # Still send to LLM for narrative content on the same page
        pages_for_llm.append(chunk)

    stage_classify.stop()
    telemetry.pages_sent_to_llm = len(pages_for_llm)

    logger.info(
        "Page classification: %d/%d pages going to LLM, %d skipped, %d local table facts",
        len(pages_for_llm),
        len(chunks_to_process),
        telemetry.pages_skipped,
        telemetry.facts_from_local_tables,
    )

    # --- Stage 3: BATCH LLM EXTRACTION ---
    stage_extract = telemetry.start_stage("llm_extraction")
    BATCH_SIZE = 6  # Pages per LLM call

    for batch_start in range(0, len(pages_for_llm), BATCH_SIZE):
        batch = pages_for_llm[batch_start : batch_start + BATCH_SIZE]

        try:
            batch_facts = await extract_facts_from_batch(
                chunks=batch,
                api_key=api_key,
                model=model,
            )
            all_extracted.extend(batch_facts)
            telemetry.facts_from_llm += len(batch_facts)
            telemetry.record_llm_call()  # Token counts tracked inside
        except Exception as e:
            err_str = str(e).lower()
            if "quota" in err_str or "resource_exhausted" in err_str:
                logger.warning(
                    "Free-tier API quota exhausted. Breaking batch loop to preserve %d extracted facts (including local table extractions).",
                    len(all_extracted),
                )
                break
            logger.warning(
                "Skipping batch pages %d-%d: %s",
                batch[0].page_number,
                batch[-1].page_number,
                e,
            )
            continue

        # Polite pacing between batches
        if batch_start + BATCH_SIZE < len(pages_for_llm):
            await asyncio.sleep(1.0)

    stage_extract.stop()

    if not all_extracted:
        return []

    # --- Stage 4: NORMALIZE & PERSIST ---
    stage_persist = telemetry.start_stage("normalization_and_storage")
    result = await process_and_store_facts(
        extracted_facts=all_extracted,
        doc_id=doc_id,
        doc_name=doc_name,
        db=db,
        registry=registry,
    )
    stage_persist.stop()

    telemetry.compute_cost_estimate(model or DEFAULT_LLM_MODEL)
    telemetry.log_summary()

    return result
