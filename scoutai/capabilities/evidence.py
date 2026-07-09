"""
Agent-callable capabilities: extract_evidence, assess_capabilities, verify_evidence.

These tools are available to candidate_agent (ADR-9). Each has:
- A single responsibility
- Strong typing
- Validated input and output
- Fail-closed behavior (ProviderError raised, never fabricated output)

Spec references: §4.2, §5, ADR-1 (confidence enum), ADR-6 (confidence on output schema).

Critical constraint (ADR-1): confidence='unknown' means NO evidence was found for this
criterion — NOT that evidence is weak. The prompt and the parsing logic both enforce this.
Never silently convert 'unknown' to 'low' or vice versa.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from scoutai.capabilities.model_router import ModelRouter, ProviderError
from scoutai.capabilities.prompts import (
    ASSESS_CAPABILITIES_SYSTEM,
    ASSESS_CAPABILITIES_USER,
    EXTRACT_EVIDENCE_SYSTEM,
    EXTRACT_EVIDENCE_USER,
    VERIFY_EVIDENCE_SYSTEM,
    VERIFY_EVIDENCE_USER,
)
from scoutai.capabilities.role_requirements import _call_llm_with_retry, _parse_json_response
from scoutai.config import ScoutAIConfig
from scoutai.schemas import (
    CapabilityAssessment,
    CapabilityHypotheses,
    EvidenceBuckets,
    EvidenceItem,
    JDRelevance,
    Relevance,
    RoleProfile,
    Rubric,
    VerificationResult,
)

logger = logging.getLogger(__name__)


# ── extract_evidence ──────────────────────────────────────────────────────────


def _parse_evidence_response(raw: str) -> dict[str, Any]:
    """Parse and validate the LLM response for extract_evidence."""
    data = _parse_json_response(raw, "extract_evidence")
    if "items" not in data:
        raise ValueError("extract_evidence response missing 'items' field")
    if not isinstance(data["items"], list):
        raise ValueError("extract_evidence 'items' must be a list")
    return data


def _build_evidence_buckets(data: dict[str, Any]) -> EvidenceBuckets:
    """Construct validated EvidenceBuckets from parsed LLM output."""
    items: list[EvidenceItem] = []
    for raw_item in data.get("items", []):
        if not isinstance(raw_item, dict):
            continue
        try:
            relevance_raw = raw_item.get("jd_relevance", {})
            if isinstance(relevance_raw, dict):
                level = relevance_raw.get("level", "medium")
                # Validate level is a valid Relevance value
                if level not in ("high", "medium", "low"):
                    level = "medium"
                rationale = str(relevance_raw.get("rationale", "Extracted from résumé"))
                if not rationale.strip():
                    rationale = "Extracted from résumé"
                jd_relevance = JDRelevance(level=level, rationale=rationale)  # type: ignore[arg-type]
            else:
                jd_relevance = JDRelevance(level="medium", rationale="Extracted from résumé")

            value = str(raw_item.get("value", "")).strip()
            source = str(raw_item.get("source", "Resume:Unknown")).strip()
            if not value or not source:
                continue

            items.append(
                EvidenceItem(
                    value=value,
                    source=source,
                    jd_relevance=jd_relevance,
                )
            )
        except Exception as e:
            logger.warning(
                "Skipping malformed evidence item",
                extra={"item": raw_item, "error": str(e)},
            )

    return EvidenceBuckets(items=items)


def extract_evidence(
    sanitized_resume: str,
    role_profile: RoleProfile,
    config: ScoutAIConfig,
    router: ModelRouter,
) -> EvidenceBuckets:
    """
    Extract JD-aligned, source-tagged evidence from a sanitized résumé.

    Agent-callable tool (§4.2). Uses fast_structured model role.
    The agent may call this multiple times (e.g., with narrower focus on
    re-extraction if evidence looks contradictory — ADR-9).

    SECURITY: sanitized_resume must have already been through screen_resume.
    This function does NOT re-screen for injection — that is enforced upstream
    by the graph (ADR-9: screen_resume runs before candidate_agent).

    Args:
        sanitized_resume: Identity-scrubbed résumé text.
        role_profile: Structured role requirements.
        config: ScoutAI configuration.
        router: Model router.

    Returns:
        EvidenceBuckets with source-tagged, JD-aligned evidence items.

    Raises:
        ProviderError: If LLM call fails after all retries.
        ValueError: If sanitized_resume is empty.
    """
    if not sanitized_resume or not sanitized_resume.strip():
        # Empty résumé → empty evidence buckets (not an error — agent handles it)
        logger.warning("extract_evidence called with empty résumé — returning empty buckets")
        return EvidenceBuckets(items=[])

    model, model_id = router.get_model("extract_evidence")

    from langchain_core.messages import HumanMessage, SystemMessage

    role_profile_json = role_profile.model_dump_json(indent=2)
    messages = [
        SystemMessage(content=EXTRACT_EVIDENCE_SYSTEM),
        HumanMessage(
            content=EXTRACT_EVIDENCE_USER.format(
                sanitized_resume=sanitized_resume,
                role_profile_json=role_profile_json,
            )
        ),
    ]

    data = _call_llm_with_retry(
        model=model,
        messages=messages,
        parse_fn=_parse_evidence_response,
        capability="extract_evidence",
        config=config,
        model_id=model_id,
        router=router,
    )

    buckets = _build_evidence_buckets(data)
    logger.info(
        "extract_evidence completed",
        extra={"evidence_count": len(buckets.items)},
    )
    return buckets


# ── assess_capabilities ───────────────────────────────────────────────────────


def _parse_capabilities_response(raw: str) -> dict[str, Any]:
    """Parse and validate the LLM response for assess_capabilities."""
    data = _parse_json_response(raw, "assess_capabilities")
    if "assessments" not in data:
        raise ValueError("assess_capabilities response missing 'assessments' field")
    if not isinstance(data["assessments"], dict):
        raise ValueError("assess_capabilities 'assessments' must be a dict")
    return data


_VALID_CONFIDENCE = {"unknown", "low", "medium", "high"}


def _build_capability_hypotheses(
    data: dict[str, Any], rubric: Rubric
) -> CapabilityHypotheses:
    """
    Construct validated CapabilityHypotheses from parsed LLM output.

    Enforces ADR-1: 'unknown' means NO evidence; 'low' means weak evidence.
    Any criterion in the rubric that is absent from the LLM response gets
    confidence='unknown' (not 'low') — it was not assessed, not assessed-as-weak.
    """
    assessments: dict[str, CapabilityAssessment] = {}
    raw_assessments: dict[str, Any] = data.get("assessments", {})

    for criterion in rubric.criteria:
        name = criterion.name
        if name in raw_assessments:
            raw = raw_assessments[name]
            if isinstance(raw, dict):
                confidence = str(raw.get("confidence", "unknown")).lower()
                if confidence not in _VALID_CONFIDENCE:
                    logger.warning(
                        "Invalid confidence value, defaulting to unknown (ADR-1)",
                        extra={"criterion": name, "raw_confidence": confidence},
                    )
                    confidence = "unknown"
                evidence_refs = [
                    str(r) for r in raw.get("evidence_refs", []) if isinstance(r, str)
                ]
                assessments[name] = CapabilityAssessment(
                    confidence=confidence,  # type: ignore[arg-type]
                    evidence_refs=evidence_refs,
                )
            else:
                assessments[name] = CapabilityAssessment(
                    confidence="unknown", evidence_refs=[]
                )
        else:
            # Criterion absent from response → unknown (ADR-1: not assessed ≠ weak)
            assessments[name] = CapabilityAssessment(
                confidence="unknown", evidence_refs=[]
            )

    return CapabilityHypotheses(assessments=assessments)


def assess_capabilities(
    evidence_buckets: EvidenceBuckets,
    rubric: Rubric,
    config: ScoutAIConfig,
    router: ModelRouter,
) -> CapabilityHypotheses:
    """
    Assess a candidate's capabilities against a rubric using extracted evidence.

    Agent-callable tool (§4.2). Uses fast_structured model role.

    Key constraint (ADR-1): confidence='unknown' and confidence='low' are DISTINCT.
    - 'unknown': the criterion was not assessed at all (no evidence to evaluate)
    - 'low': the criterion WAS assessed, but evidence is weak

    This distinction is preserved through parsing and must not be lost downstream.

    Args:
        evidence_buckets: Extracted evidence from the résumé.
        rubric: The evaluation rubric.
        config: ScoutAI configuration.
        router: Model router.

    Returns:
        CapabilityHypotheses with per-criterion confidence + evidence_refs.

    Raises:
        ProviderError: If LLM call fails after all retries.
    """
    if not rubric.criteria:
        raise ValueError("Rubric has no criteria — cannot assess capabilities")

    model, model_id = router.get_model("assess_capabilities")

    from langchain_core.messages import HumanMessage, SystemMessage

    evidence_json = evidence_buckets.model_dump_json(indent=2)
    rubric_json = rubric.model_dump_json(indent=2)
    messages = [
        SystemMessage(content=ASSESS_CAPABILITIES_SYSTEM),
        HumanMessage(
            content=ASSESS_CAPABILITIES_USER.format(
                evidence_json=evidence_json,
                rubric_json=rubric_json,
            )
        ),
    ]

    data = _call_llm_with_retry(
        model=model,
        messages=messages,
        parse_fn=_parse_capabilities_response,
        capability="assess_capabilities",
        config=config,
        model_id=model_id,
        router=router,
    )

    hypotheses = _build_capability_hypotheses(data, rubric)
    logger.info(
        "assess_capabilities completed",
        extra={
            "criteria_count": len(hypotheses.assessments),
            "unknown_count": sum(
                1 for a in hypotheses.assessments.values()
                if a.confidence == "unknown"
            ),
        },
    )
    return hypotheses


# ── verify_evidence ───────────────────────────────────────────────────────────


def _parse_verification_response(raw: str) -> dict[str, Any]:
    """Parse and validate the LLM response for verify_evidence."""
    data = _parse_json_response(raw, "verify_evidence")
    if "verdict" not in data:
        raise ValueError("verify_evidence response missing 'verdict' field")
    if data["verdict"] not in ("sufficient", "insufficient"):
        raise ValueError(
            f"verify_evidence verdict must be 'sufficient' or 'insufficient', "
            f"got {data['verdict']!r}"
        )
    return data


def _build_verification_result(data: dict[str, Any]) -> VerificationResult:
    """Construct validated VerificationResult from parsed LLM output."""
    verdict = data["verdict"]  # already validated
    per_gap = {}
    raw_gaps = data.get("per_gap_reasoning", {})
    if isinstance(raw_gaps, dict):
        for k, v in raw_gaps.items():
            per_gap[str(k)] = str(v)
    return VerificationResult(verdict=verdict, per_gap_reasoning=per_gap)


def verify_evidence(
    capabilities: CapabilityHypotheses,
    rubric: Rubric,
    config: ScoutAIConfig,
    router: ModelRouter,
) -> VerificationResult:
    """
    Verify whether the evidence gathered is sufficient to make a recommendation.

    Agent-callable tool (§4.2). Uses fast_structured model role.
    Returns 'sufficient' or 'insufficient' with per-criterion reasoning.

    The agent uses this verdict to decide whether to:
    - Call ask_candidate (if insufficient and ask_candidate is still available)
    - Call finalize_candidate (if sufficient, or if uncertainty is acceptable)

    Args:
        capabilities: Current capability assessments.
        rubric: The evaluation rubric.
        config: ScoutAI configuration.
        router: Model router.

    Returns:
        VerificationResult with verdict + per-gap reasoning.

    Raises:
        ProviderError: If LLM call fails after all retries.
    """
    model, model_id = router.get_model("verify_evidence")

    from langchain_core.messages import HumanMessage, SystemMessage

    capabilities_json = capabilities.model_dump_json(indent=2)
    rubric_json = rubric.model_dump_json(indent=2)
    messages = [
        SystemMessage(content=VERIFY_EVIDENCE_SYSTEM),
        HumanMessage(
            content=VERIFY_EVIDENCE_USER.format(
                capabilities_json=capabilities_json,
                rubric_json=rubric_json,
            )
        ),
    ]

    data = _call_llm_with_retry(
        model=model,
        messages=messages,
        parse_fn=_parse_verification_response,
        capability="verify_evidence",
        config=config,
        model_id=model_id,
        router=router,
    )

    result = _build_verification_result(data)
    logger.info(
        "verify_evidence completed",
        extra={"verdict": result.verdict},
    )
    return result
