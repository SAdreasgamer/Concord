"""
Pipeline telemetry — tracks API calls, cost, and performance metrics.

Provides full visibility into what the pipeline did and how efficiently:
- How many pages were skipped vs processed
- How many API calls were made
- Estimated API cost
- Processing time per stage
- Local vs LLM extraction breakdown

This answers the interviewer question: "How efficient is your system?"
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class StageTiming:
    """Timing for a single pipeline stage."""
    stage_name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0

    def start(self) -> None:
        self.start_time = time.time()

    def stop(self) -> None:
        self.end_time = time.time()
        self.duration_seconds = self.end_time - self.start_time


@dataclass
class PipelineTelemetry:
    """
    Accumulates metrics across a document processing run.
    
    Provides a clear answer to "How efficient is your system?" with
    concrete numbers on API savings, cost, and performance.
    """

    # --- Page Classification Stats ---
    total_pages: int = 0
    pages_sent_to_llm: int = 0
    pages_skipped: int = 0
    pages_with_local_tables: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)

    # --- Extraction Stats ---
    llm_api_calls: int = 0
    llm_total_input_tokens: int = 0
    llm_total_output_tokens: int = 0
    facts_from_llm: int = 0
    facts_from_local_tables: int = 0
    total_facts: int = 0

    # --- Matching & Judging Stats ---
    candidate_pairs_found: int = 0
    judge_api_calls: int = 0
    relationships_discovered: int = 0

    # --- Cost Estimation ---
    estimated_cost_usd: float = 0.0

    # --- Timing ---
    stage_timings: list[StageTiming] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    _start_time: float = 0.0

    def start(self) -> None:
        """Mark the start of the pipeline run."""
        self._start_time = time.time()

    def stop(self) -> None:
        """Mark the end of the pipeline run."""
        self.total_duration_seconds = time.time() - self._start_time

    def start_stage(self, stage_name: str) -> StageTiming:
        """Start timing a pipeline stage."""
        timing = StageTiming(stage_name=stage_name)
        timing.start()
        self.stage_timings.append(timing)
        return timing

    def record_page_skip(self, reason: str) -> None:
        """Record a page being skipped."""
        self.pages_skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1

    def record_llm_call(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Record an LLM API call with token counts."""
        self.llm_api_calls += 1
        self.llm_total_input_tokens += input_tokens
        self.llm_total_output_tokens += output_tokens

    def compute_cost_estimate(self, model: str = "gemini/gemini-3.6-flash") -> float:
        """
        Estimate API cost based on token usage.
        
        Pricing (as of 2025):
        - Gemini Flash: $0.075/1M input tokens, $0.30/1M output tokens
        - GPT-4o: $2.50/1M input, $10/1M output
        - Claude 3.5 Sonnet: $3/1M input, $15/1M output
        """
        model_lower = model.lower()
        
        if "gemini" in model_lower and "flash" in model_lower:
            input_cost_per_m = 0.075
            output_cost_per_m = 0.30
        elif "gemini" in model_lower and "pro" in model_lower:
            input_cost_per_m = 1.25
            output_cost_per_m = 5.00
        elif "gpt-4o" in model_lower:
            input_cost_per_m = 2.50
            output_cost_per_m = 10.00
        elif "claude" in model_lower:
            input_cost_per_m = 3.00
            output_cost_per_m = 15.00
        else:
            input_cost_per_m = 0.50  # Conservative default
            output_cost_per_m = 1.50
        
        input_cost = (self.llm_total_input_tokens / 1_000_000) * input_cost_per_m
        output_cost = (self.llm_total_output_tokens / 1_000_000) * output_cost_per_m
        self.estimated_cost_usd = input_cost + output_cost
        return self.estimated_cost_usd

    @property
    def pages_saved_percentage(self) -> float:
        """Percentage of pages that didn't need LLM processing."""
        if self.total_pages == 0:
            return 0.0
        return (self.pages_skipped / self.total_pages) * 100

    @property
    def local_extraction_ratio(self) -> float:
        """Percentage of facts extracted locally without LLM."""
        if self.total_facts == 0:
            return 0.0
        return (self.facts_from_local_tables / self.total_facts) * 100

    @property
    def total_api_calls(self) -> int:
        """Total API calls (extraction + judging)."""
        return self.llm_api_calls + self.judge_api_calls

    def to_dict(self) -> dict:
        """Serialize telemetry to a dictionary for API responses."""
        self.total_facts = self.facts_from_llm + self.facts_from_local_tables
        
        return {
            "pipeline_efficiency": {
                "total_pages": self.total_pages,
                "pages_sent_to_llm": self.pages_sent_to_llm,
                "pages_skipped": self.pages_skipped,
                "pages_saved_pct": round(self.pages_saved_percentage, 1),
                "skip_reasons": self.skip_reasons,
            },
            "extraction": {
                "facts_from_llm": self.facts_from_llm,
                "facts_from_local_tables": self.facts_from_local_tables,
                "total_facts": self.total_facts,
                "local_extraction_ratio_pct": round(self.local_extraction_ratio, 1),
            },
            "api_usage": {
                "extraction_api_calls": self.llm_api_calls,
                "judge_api_calls": self.judge_api_calls,
                "total_api_calls": self.total_api_calls,
                "total_input_tokens": self.llm_total_input_tokens,
                "total_output_tokens": self.llm_total_output_tokens,
                "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            },
            "matching": {
                "candidate_pairs": self.candidate_pairs_found,
                "relationships_discovered": self.relationships_discovered,
            },
            "performance": {
                "total_duration_seconds": round(self.total_duration_seconds, 2),
                "stages": [
                    {
                        "name": s.stage_name,
                        "duration_seconds": round(s.duration_seconds, 2),
                    }
                    for s in self.stage_timings
                ],
            },
        }

    def log_summary(self) -> None:
        """Log a human-readable summary of the pipeline run."""
        self.total_facts = self.facts_from_llm + self.facts_from_local_tables
        
        logger.info(
            "\n📊 Pipeline Telemetry Summary\n"
            "  Pages: %d total → %d to LLM, %d skipped (%.0f%% saved)\n"
            "  Facts: %d total → %d from LLM, %d from local tables (%.0f%% local)\n"
            "  API Calls: %d extraction + %d judging = %d total\n"
            "  Est. Cost: $%.4f\n"
            "  Duration: %.1fs",
            self.total_pages, self.pages_sent_to_llm, self.pages_skipped, self.pages_saved_percentage,
            self.total_facts, self.facts_from_llm, self.facts_from_local_tables, self.local_extraction_ratio,
            self.llm_api_calls, self.judge_api_calls, self.total_api_calls,
            self.estimated_cost_usd,
            self.total_duration_seconds,
        )


class TelemetryTracker:
    """
    In-memory registry of telemetry across multiple document ingestion runs.
    Provides cumulative efficiency metrics and comparative savings over naive baseline.
    """

    def __init__(self) -> None:
        self.runs: list[dict] = []
        self.cumulative_pages: int = 0
        self.cumulative_skipped_pages: int = 0
        self.cumulative_api_calls: int = 0
        self.cumulative_local_facts: int = 0
        self.cumulative_llm_facts: int = 0
        self.cumulative_cost_usd: float = 0.0

    def record_run(self, doc_id: str, doc_name: str, telemetry: PipelineTelemetry) -> dict:
        """Record a completed document run."""
        data = telemetry.to_dict()
        data["doc_id"] = doc_id
        data["doc_name"] = doc_name
        data["timestamp"] = time.time()
        
        self.runs.append(data)
        self.cumulative_pages += telemetry.total_pages
        self.cumulative_skipped_pages += telemetry.pages_skipped
        self.cumulative_api_calls += telemetry.total_api_calls
        self.cumulative_local_facts += telemetry.facts_from_local_tables
        self.cumulative_llm_facts += telemetry.facts_from_llm
        self.cumulative_cost_usd += telemetry.estimated_cost_usd

        return data

    def get_summary(self) -> dict:
        """Return cumulative metrics and comparison against a naive pipeline."""
        total_facts = self.cumulative_local_facts + self.cumulative_llm_facts
        local_ratio = round((self.cumulative_local_facts / total_facts * 100), 1) if total_facts else 0.0
        pages_saved_pct = round((self.cumulative_skipped_pages / self.cumulative_pages * 100), 1) if self.cumulative_pages else 0.0

        # Naive baseline: 1 extraction API call per page
        naive_extraction_calls = self.cumulative_pages
        actual_extraction_calls = sum(r.get("api_usage", {}).get("extraction_api_calls", 0) for r in self.runs)
        saved_calls = max(0, naive_extraction_calls - actual_extraction_calls)
        savings_factor = round(naive_extraction_calls / max(actual_extraction_calls, 1), 1) if actual_extraction_calls else 1.0

        # Cost baseline: naive would cost ~7x more tokens
        naive_cost_usd = round(self.cumulative_cost_usd * max(savings_factor, 1.0), 5)
        cost_saved_usd = round(max(0.0, naive_cost_usd - self.cumulative_cost_usd), 5)

        return {
            "total_runs": len(self.runs),
            "cumulative_pages": self.cumulative_pages,
            "cumulative_skipped_pages": self.cumulative_skipped_pages,
            "pages_saved_pct": pages_saved_pct,
            "cumulative_api_calls": self.cumulative_api_calls,
            "cumulative_local_facts": self.cumulative_local_facts,
            "cumulative_llm_facts": self.cumulative_llm_facts,
            "total_facts": total_facts,
            "local_extraction_ratio_pct": local_ratio,
            "cumulative_cost_usd": round(self.cumulative_cost_usd, 5),
            "naive_baseline": {
                "naive_extraction_calls": naive_extraction_calls,
                "actual_extraction_calls": actual_extraction_calls,
                "calls_saved": saved_calls,
                "reduction_factor": f"{savings_factor}x",
                "estimated_naive_cost_usd": naive_cost_usd,
                "cost_saved_usd": cost_saved_usd,
            },
            "recent_runs": self.runs[-10:],
        }

    def get_run(self, doc_id: str) -> Optional[dict]:
        """Find telemetry for a specific document ID."""
        for run in reversed(self.runs):
            if run.get("doc_id") == doc_id:
                return run
        return None

    def clear(self) -> None:
        """Reset all tracked metrics."""
        self.runs.clear()
        self.cumulative_pages = 0
        self.cumulative_skipped_pages = 0
        self.cumulative_api_calls = 0
        self.cumulative_local_facts = 0
        self.cumulative_llm_facts = 0
        self.cumulative_cost_usd = 0.0


# Global tracker instance
global_telemetry = TelemetryTracker()

