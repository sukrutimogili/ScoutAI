"""
candidate_agent — bounded ReAct tool-calling harness (ADR-9, §3.3).

This is the agentic node inside the LangGraph outer graph. It wraps all S5/S6
tool functions in a bounded loop, enforcing:

1. Hard tool allowlist (§7.4): only the 7 approved tools are callable.
   The agent cannot invoke screen_resume, run_fairness_probe, human_review,
   schedule, or any fixed-node capability — regardless of what it "decides".

2. max_iterations budget (§3.3): 8 tool calls per candidate pass.
   If the budget is exhausted without finalize_candidate being called,
   the harness force-finalizes with recommendation='hold' (fail-closed, §3.3).

3. ask_candidate single-use enforcement (ADR-3):
   After the first ask_candidate call, it is REMOVED from the available tool
   list at the harness level — not just by prompt instruction.
   interview_rounds in CandidateState tracks this at the state level.

4. Every tool call is logged to trajectory (ADR-8).
   The harness does this — not the agent, not the tools themselves.

Architecture note:
- The agent uses a LangChain tool-calling loop with the fast_structured model.
- The "system prompt" encodes policy; the harness enforces hard limits.
- finalize_candidate is the ONLY exit from the loop (no implicit fall-through).
- If the LLM returns a tool call not in the allowlist, the harness rejects it
  and logs a warning — it does not execute unknown tools.

Spec references: §3.2, §3.3, §4.2, ADR-3, ADR-8, ADR-9.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from scoutai.audit.metrics import MetricsCollector
from scoutai.capabilities.evidence import (
    assess_capabilities,
    extract_evidence,
    verify_evidence,
)
from scoutai.capabilities.interview import (
    AskCandidateResult,
    create_ask_candidate_interrupt,
    finalize_candidate,
    generate_interview_questions,
    reevaluate_candidate,
)
from scoutai.capabilities.model_router import ModelRouter
from scoutai.config import ScoutAIConfig
from scoutai.schemas import (
    CandidateState,
    CapabilityHypotheses,
    EvidenceBuckets,
    Rubric,
    RoleProfile,
    ShortlistEntry,
    TrajectoryEntry,
    VerificationResult,
)

logger = logging.getLogger(__name__)

# ── Exception classification ──────────────────────────────────────────────────

# TPM threshold: if retry-after is <= this many seconds we treat it as a
# per-minute token limit (short wait, bounded retry is worthwhile).
# Above this threshold we treat it as a daily quota (TPD) — don't block,
# trip the circuit breaker and fail-close immediately.
_TPM_RETRY_AFTER_THRESHOLD_S = 120  # 2 minutes


class _RateLimitKind:
    """Classification result from classify_llm_exception."""
    __slots__ = ("is_rate_limit", "is_tpd", "retry_after_s", "provider", "detail")

    def __init__(
        self,
        *,
        is_rate_limit: bool = False,
        is_tpd: bool = False,
        retry_after_s: float = 0.0,
        provider: str = "",
        detail: str = "",
    ) -> None:
        self.is_rate_limit = is_rate_limit
        self.is_tpd = is_tpd          # True = daily quota (long wait), False = TPM (short wait)
        self.retry_after_s = retry_after_s
        self.provider = provider
        self.detail = detail


def classify_llm_exception(exc: Exception) -> _RateLimitKind:
    """
    Classify an exception from a bound LLM invoke() call.

    Detects:
    - Groq:   groq.RateLimitError  (HTTP 429)
    - Gemini: google.api_core.exceptions.ResourceExhausted (HTTP 429 / gRPC RESOURCE_EXHAUSTED)

    Distinguishes TPM (tokens-per-minute, retry-after <= 2 min) from TPD
    (tokens-per-day, retry-after > 2 min).

    Returns a _RateLimitKind with is_rate_limit=False for non-quota exceptions.
    """
    exc_type = type(exc).__name__
    exc_str = str(exc)

    # ── Groq ──────────────────────────────────────────────────────────────────
    # groq.RateLimitError is raised directly by langchain-groq on HTTP 429.
    try:
        import groq
        if isinstance(exc, groq.RateLimitError):
            return _parse_rate_limit("groq", exc_str)
    except ImportError:
        pass

    # ── Gemini / Google ───────────────────────────────────────────────────────
    # google.api_core.exceptions.ResourceExhausted wraps gRPC RESOURCE_EXHAUSTED.
    # langchain-google-genai re-raises it as-is.
    try:
        from google.api_core.exceptions import ResourceExhausted
        if isinstance(exc, ResourceExhausted):
            return _parse_rate_limit("gemini", exc_str)
    except ImportError:
        pass

    # ── Fallback: string-match for when the exception is wrapped ──────────────
    # LangChain sometimes wraps provider errors; check message text as a safety net.
    lower = exc_str.lower()
    if "rate limit" in lower or "ratelimit" in lower or "resource_exhausted" in lower or (
        "429" in exc_str and ("token" in lower or "quota" in lower)
    ):
        provider = "gemini" if "generativelanguage" in lower or "gemini" in lower else "groq"
        return _parse_rate_limit(provider, exc_str)

    return _RateLimitKind()


def _parse_rate_limit(provider: str, exc_str: str) -> _RateLimitKind:
    """
    Parse retry-after seconds out of the exception message and classify
    the limit as TPM (short) or TPD (long / daily quota).

    Groq messages contain e.g. "Please try again in 9m31.104s" or "in 205ms".
    Gemini messages contain "retry_delay { seconds: 17 }".
    """
    import re

    retry_after_s = 0.0

    # Groq: "Please try again in 9m31.104s" / "in 30s" / "in 205ms"
    m = re.search(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)(s|ms)", exc_str)
    if m:
        minutes = float(m.group(1) or 0)
        amount  = float(m.group(2))
        unit    = m.group(3)
        retry_after_s = minutes * 60 + (amount / 1000 if unit == "ms" else amount)

    # Gemini: "retry_delay { seconds: 17 }"
    if retry_after_s == 0.0:
        m2 = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", exc_str)
        if m2:
            retry_after_s = float(m2.group(1))

    # Determine TPM vs TPD
    is_tpd = retry_after_s > _TPM_RETRY_AFTER_THRESHOLD_S

    # Truncate detail to avoid storing huge stack traces
    detail = f"{provider} 429: {exc_str[:300]}"

    return _RateLimitKind(
        is_rate_limit=True,
        is_tpd=is_tpd,
        retry_after_s=retry_after_s,
        provider=provider,
        detail=detail,
    )

# ── Agent system prompt ───────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """You are a hiring evaluation agent. Your job is to assess a candidate against a job rubric and produce a final recommendation.

You have access to these tools (use them in a logical order):
1. extract_evidence — extract evidence from the résumé relevant to the role
2. assess_capabilities — score the candidate against each rubric criterion
3. verify_evidence — check if evidence is sufficient for a recommendation
4. generate_interview_questions — generate questions targeting evidence gaps (use ONCE only)
5. ask_candidate — ask ONE clarifying question (available once; will be removed after use)
6. reevaluate_candidate — update scores based on the interview answer
7. finalize_candidate — REQUIRED: call this to end your evaluation with a recommendation

Policy (non-negotiable):
- Never finalize a criterion at 'unknown' confidence without first considering ask_candidate
- Do not call generate_interview_questions more than once
- ask_candidate is limited to 1 use — choose your question carefully
- You MUST call finalize_candidate to end your turn — there is no implicit finish
- Recommendation options: reject | hold | interview | strong_interview
- Be honest about uncertainty — 'hold' is valid when evidence is insufficient

Your budget: you have a limited number of tool calls. Use them efficiently."""


# ── Tool dispatcher ───────────────────────────────────────────────────────────


class AgentHarness:
    """
    Bounded ReAct agent harness for candidate_agent.

    Manages the tool-calling loop with hard enforcement of:
    - Tool allowlist
    - max_iterations budget
    - ask_candidate single-use (ADR-3)
    - force-finalize on budget exhaustion (§3.3)
    - trajectory logging for every tool call (ADR-8)

    Usage:
        harness = AgentHarness(config, router)
        result = harness.run(candidate, role_profile, rubric)
        # result is (updated_candidate, shortlist_entry, trajectory_entries)
    """

    # Immutable allowlist — the agent can ONLY call these tools (§7.4)
    TOOL_ALLOWLIST = frozenset({
        "extract_evidence",
        "assess_capabilities",
        "verify_evidence",
        "generate_interview_questions",
        "ask_candidate",
        "reevaluate_candidate",
        "finalize_candidate",
    })

    def __init__(self, config: ScoutAIConfig, router: ModelRouter) -> None:
        self._config = config
        self._router = router
        self._max_iterations = config.agent.max_iterations
        self._schema_version = config.schema_version
        self._tool_versions = config.tool_versions
        self._prompt_version = config.prompt_versions.get("extract_evidence", "1.0.0")

    def run(
        self,
        candidate: CandidateState,
        role_profile: RoleProfile,
        rubric: Rubric,
    ) -> tuple[CandidateState, Optional[ShortlistEntry], list[TrajectoryEntry]]:
        """
        Run the candidate_agent loop for one candidate.

        Returns:
            (updated_candidate, shortlist_entry, new_trajectory_entries)
            shortlist_entry is None if finalize_candidate was never called
            (which shouldn't happen — force-finalize ensures it always is).
        """
        trajectory: list[TrajectoryEntry] = []
        available_tools = set(self.TOOL_ALLOWLIST)  # mutable per-run copy

        # Per-run state
        evidence_buckets: Optional[EvidenceBuckets] = None
        capabilities: Optional[CapabilityHypotheses] = None
        verification: Optional[VerificationResult] = None
        shortlist_entry: Optional[ShortlistEntry] = None
        updated_candidate = candidate.model_copy(deep=True)

        # LLM for the agent loop
        model, model_id = self._router.get_model("extract_evidence")  # fast_structured

        # Build initial message history
        messages: list[Any] = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=self._build_task_message(candidate, role_profile, rubric)),
        ]

        for iteration in range(self._max_iterations):
            _iter_start = time.monotonic()
            _last_iter_elapsed_ms = 0  # track for budget-exhausted fallback

            # Remove ask_candidate after it's been used (ADR-3)
            if updated_candidate.interview_rounds >= 1:
                available_tools.discard("ask_candidate")

            # Get tool definitions for this iteration
            tools_json = self._build_tools_description(available_tools)

            # Invoke the LLM with current message history + available tools
            logger.debug(
                "Agent iteration",
                extra={
                    "iteration": iteration + 1,
                    "candidate_id": candidate.candidate_id,
                    "available_tools": sorted(available_tools),
                },
            )

            try:
                # Bind tools to the model and invoke
                bound_model = model.bind_tools(
                    self._get_langchain_tools(available_tools, updated_candidate, role_profile, rubric)
                )
                response = bound_model.invoke(messages)
            except Exception as e:
                llm_call_elapsed_ms = int((time.monotonic() - _iter_start) * 1000)
                kind = classify_llm_exception(e)
                error_detail = (
                    f"{type(e).__name__}: {str(e)[:300]} (iteration={iteration + 1})"
                )

                if kind.is_rate_limit and not kind.is_tpd:
                    # ── TPM (per-minute): short retry-after, worth waiting ──────
                    wait_s = kind.retry_after_s if kind.retry_after_s > 0 else 5.0
                    logger.warning(
                        "Agent LLM call hit TPM rate limit — retrying after backoff",
                        extra={
                            "iteration": iteration + 1,
                            "retry_after_s": wait_s,
                            "provider": kind.provider,
                        },
                    )
                    time.sleep(wait_s)
                    continue  # retry this iteration

                if kind.is_rate_limit and kind.is_tpd:
                    # ── TPD (daily quota): don't block, trip CB and fail-close ──
                    logger.error(
                        "Agent LLM call hit TPD quota — tripping circuit breaker",
                        extra={
                            "provider": kind.provider,
                            "retry_after_s": kind.retry_after_s,
                            "candidate_id": candidate.candidate_id,
                        },
                    )
                    self._router.circuit_breaker.record_failure(kind.provider)
                    return self._force_finalize(
                        updated_candidate, rubric, trajectory, capabilities,
                        elapsed_ms=llm_call_elapsed_ms,
                        error_detail=error_detail,
                        status="failed_closed_quota_exhausted",
                    )

                # ── Non-quota exception ────────────────────────────────────────
                logger.error(
                    "Agent LLM call failed",
                    extra={
                        "iteration": iteration + 1,
                        "error": str(e),
                        "exc_type": type(e).__name__,
                    },
                )
                return self._force_finalize(
                    updated_candidate, rubric, trajectory, capabilities,
                    elapsed_ms=llm_call_elapsed_ms,
                    error_detail=error_detail,
                    status="failed_closed",
                )

            messages.append(response)

            # Check if there are tool calls in the response
            tool_calls = getattr(response, "tool_calls", []) or []

            if not tool_calls:
                # LLM returned a response but called no tools and didn't call
                # finalize_candidate — treat as a failed iteration.
                elapsed_ms = int((time.monotonic() - _iter_start) * 1000)
                error_detail = (
                    f"NoToolCalls: LLM returned no tool calls without calling "
                    f"finalize_candidate (iteration={iteration + 1})"
                )
                logger.warning(
                    "Agent returned no tool calls without calling finalize_candidate",
                    extra={"iteration": iteration + 1, "elapsed_ms": elapsed_ms},
                )
                return self._force_finalize(
                    updated_candidate, rubric, trajectory, capabilities,
                    elapsed_ms=elapsed_ms,
                    error_detail=error_detail,
                    status="failed_closed",
                )

            # Execute each tool call
            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "") if isinstance(tool_call, dict) else getattr(tool_call, "name", "")
                tool_args = tool_call.get("args", {}) if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
                tool_call_id = tool_call.get("id", f"call_{iteration}") if isinstance(tool_call, dict) else getattr(tool_call, "id", f"call_{iteration}")

                # Security: reject tools not in allowlist
                if tool_name not in self.TOOL_ALLOWLIST:
                    logger.warning(
                        "Agent attempted to call tool not in allowlist — REJECTED",
                        extra={"tool": tool_name, "candidate_id": candidate.candidate_id},
                    )
                    messages.append(ToolMessage(
                        content=f"ERROR: Tool '{tool_name}' is not available.",
                        tool_call_id=tool_call_id,
                    ))
                    continue

                if tool_name not in available_tools:
                    logger.warning(
                        "Agent attempted to call unavailable tool — REJECTED",
                        extra={"tool": tool_name, "reason": "ask_candidate already used (ADR-3)"},
                    )
                    messages.append(ToolMessage(
                        content=f"ERROR: Tool '{tool_name}' has already been used and is no longer available.",
                        tool_call_id=tool_call_id,
                    ))
                    continue

                # Execute the tool
                tool_result, traj_entry = self._execute_tool(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    candidate=updated_candidate,
                    role_profile=role_profile,
                    rubric=rubric,
                    evidence_buckets=evidence_buckets,
                    capabilities=capabilities,
                )
                trajectory.append(traj_entry)

                # Update running state based on tool result
                if tool_name == "extract_evidence" and isinstance(tool_result, EvidenceBuckets):
                    evidence_buckets = tool_result
                    updated_candidate = updated_candidate.model_copy(
                        update={"evidence_buckets": tool_result.items}
                    )
                elif tool_name == "assess_capabilities" and isinstance(tool_result, CapabilityHypotheses):
                    capabilities = tool_result
                    updated_candidate = updated_candidate.model_copy(
                        update={"capabilities": tool_result.assessments}
                    )
                elif tool_name == "verify_evidence" and isinstance(tool_result, VerificationResult):
                    verification = tool_result
                    updated_candidate = updated_candidate.model_copy(
                        update={"verification_verdict": tool_result.verdict}
                    )
                elif tool_name == "ask_candidate":
                    # Interrupt — mark as used (ADR-3)
                    updated_candidate = updated_candidate.model_copy(
                        update={"interview_rounds": 1}
                    )
                    available_tools.discard("ask_candidate")
                    # tool_result is the interrupt payload dict
                elif tool_name == "reevaluate_candidate":
                    from scoutai.schemas import UpdatedScoreCard
                    if hasattr(tool_result, "scorecard"):
                        updated_candidate = updated_candidate.model_copy(
                            update={
                                "scorecard": tool_result.scorecard,
                                "capabilities": tool_result.capabilities,
                            }
                        )
                        capabilities = CapabilityHypotheses(assessments=tool_result.capabilities)
                elif tool_name == "finalize_candidate" and isinstance(tool_result, ShortlistEntry):
                    shortlist_entry = tool_result
                    updated_candidate = updated_candidate.model_copy(
                        update={
                            "recommendation": tool_result.recommendation,
                            "rationale": str(tool_args.get("rationale", "")),
                            "finalized": True,
                            "remaining_uncertainties": tool_result.remaining_uncertainties,
                            # Copy strengths so candidate_list/candidate_review can render them
                            "strengths": tool_result.strengths,
                        }
                    )

                # Add tool result to message history
                tool_result_str = self._serialize_tool_result(tool_name, tool_result)
                messages.append(ToolMessage(
                    content=tool_result_str,
                    tool_call_id=tool_call_id,
                ))

                # Track elapsed time for budget-exhausted fallback
                _last_iter_elapsed_ms = int((time.monotonic() - _iter_start) * 1000)

                # If finalize was called, we're done
                if tool_name == "finalize_candidate":
                    logger.info(
                        "candidate_agent finalized",
                        extra={
                            "candidate_id": candidate.candidate_id,
                            "recommendation": tool_result.recommendation if isinstance(tool_result, ShortlistEntry) else "unknown",
                            "iterations": iteration + 1,
                        },
                    )
                    return updated_candidate, shortlist_entry, trajectory

        # ── Budget exhausted without finalize_candidate ────────────────────────
        logger.warning(
            "candidate_agent budget exhausted — force-finalizing with 'hold' (§3.3)",
            extra={
                "candidate_id": candidate.candidate_id,
                "iterations_used": self._max_iterations,
            },
        )
        return self._force_finalize(
            updated_candidate, rubric, trajectory, capabilities,
            elapsed_ms=_last_iter_elapsed_ms,
            error_detail=(
                f"BudgetExhausted: {self._max_iterations} tool calls used without "
                f"finalize_candidate (candidate={candidate.candidate_id})"
            ),
            status="failed_closed",
        )

    def _force_finalize(
        self,
        candidate: CandidateState,
        rubric: Rubric,
        trajectory: list[TrajectoryEntry],
        capabilities: Optional[CapabilityHypotheses],
        *,
        elapsed_ms: int = 0,
        error_detail: Optional[str] = None,
        status: str = "failed_closed",
    ) -> tuple[CandidateState, ShortlistEntry, list[TrajectoryEntry]]:
        """
        Force-finalize a candidate when the loop exits without finalize_candidate.

        Per §3.3: recommendation='hold', remaining_uncertainties populated from
        last known state. This is a fail-closed path — never silent.

        Args:
            elapsed_ms:   Real wall-clock time of the failing operation in ms.
            error_detail: Human-readable cause: '<ExcType>: <msg> (iteration=N)'.
            status:       Trajectory status — 'failed_closed' for generic failures,
                          'failed_closed_quota_exhausted' for TPD quota hits.
        """
        remaining = [
            name for name, cap in (capabilities.assessments if capabilities else {}).items()
            if cap.confidence in ("unknown", "low")
        ]
        if not remaining:
            remaining = [c.name for c in rubric.criteria]  # all uncertain

        force_rec = self._config.agent.force_finalize_recommendation
        rationale = (
            f"Evaluation budget exhausted after {self._max_iterations} tool calls. "
            f"Insufficient evidence gathered to make a confident recommendation. "
            f"Force-finalized with '{force_rec}'."
        )
        if error_detail:
            rationale = f"{rationale} Cause: {error_detail}"

        try:
            entry = finalize_candidate(
                candidate=candidate,
                recommendation=force_rec,  # type: ignore[arg-type]
                rationale=rationale,
                rubric=rubric,
                config=self._config,
                remaining_uncertainties=remaining,
            )
        except Exception as e:
            # If even force-finalize fails, create a minimal safe entry
            logger.error("Force-finalize failed", extra={"error": str(e)})
            from scoutai.schemas import ShortlistEntry
            entry = ShortlistEntry(
                candidate=candidate.candidate_id,
                recommendation="hold",
                weighted_score=0.0,
                remaining_uncertainties=remaining,
                evidence_refs=[],
            )

        # Record force-finalize in trajectory with real latency and error detail
        traj = self._make_trajectory_entry(
            node="candidate_agent",
            tool_used="force_finalize",
            latency_ms=elapsed_ms,
            model=None,
            status=status,
            error_detail=error_detail,
        )
        trajectory.append(traj)

        updated = candidate.model_copy(
            update={
                "recommendation": force_rec,
                "rationale": rationale,
                "finalized": True,
                "remaining_uncertainties": remaining,
                "strengths": entry.strengths,
            }
        )
        return updated, entry, trajectory

    def _execute_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        candidate: CandidateState,
        role_profile: RoleProfile,
        rubric: Rubric,
        evidence_buckets: Optional[EvidenceBuckets],
        capabilities: Optional[CapabilityHypotheses],
    ) -> tuple[Any, TrajectoryEntry]:
        """
        Execute a single tool call and return (result, trajectory_entry).

        All tool results are wrapped in try/except — tool failures are logged
        and a safe error value is returned rather than crashing the loop.
        """
        start_time = time.monotonic()
        status = "success"
        error_detail: Optional[str] = None
        result: Any = None

        try:
            if tool_name == "extract_evidence":
                result = extract_evidence(
                    sanitized_resume=candidate.sanitized_resume,
                    role_profile=role_profile,
                    config=self._config,
                    router=self._router,
                )
            elif tool_name == "assess_capabilities":
                buckets = evidence_buckets or EvidenceBuckets(items=[])
                result = assess_capabilities(
                    evidence_buckets=buckets,
                    rubric=rubric,
                    config=self._config,
                    router=self._router,
                )
            elif tool_name == "verify_evidence":
                caps = capabilities or CapabilityHypotheses(assessments={})
                result = verify_evidence(
                    capabilities=caps,
                    rubric=rubric,
                    config=self._config,
                    router=self._router,
                )
            elif tool_name == "generate_interview_questions":
                caps = capabilities or CapabilityHypotheses(assessments={})
                result = generate_interview_questions(
                    capabilities=caps,
                    rubric=rubric,
                    config=self._config,
                    router=self._router,
                )
            elif tool_name == "ask_candidate":
                # Create interrupt payload — actual interrupt handled by LangGraph
                from scoutai.capabilities.interview import InterviewQuestion
                question_text = str(tool_args.get("question", ""))
                criterion = str(tool_args.get("target_criterion", "unknown"))
                question = InterviewQuestion(
                    question=question_text,
                    target_criterion=criterion,
                    rationale=str(tool_args.get("rationale", "")),
                    priority_score=float(tool_args.get("priority_score", 1.0)),
                )
                result = create_ask_candidate_interrupt(question, candidate.candidate_id)
            elif tool_name == "reevaluate_candidate":
                result = reevaluate_candidate(
                    candidate=candidate,
                    question=str(tool_args.get("question", "")),
                    answer=str(tool_args.get("answer", "")),
                    rubric=rubric,
                    config=self._config,
                    router=self._router,
                )
            elif tool_name == "finalize_candidate":
                rec = str(tool_args.get("recommendation", "hold"))
                rationale = str(tool_args.get("rationale", "Agent finalized evaluation"))
                remaining = list(tool_args.get("remaining_uncertainties", []))
                result = finalize_candidate(
                    candidate=candidate,
                    recommendation=rec,  # type: ignore[arg-type]
                    rationale=rationale,
                    rubric=rubric,
                    config=self._config,
                    remaining_uncertainties=remaining if remaining else None,
                )
        except Exception as e:
            error_detail = f"{type(e).__name__}: {str(e)[:300]}"
            logger.error(
                "Tool execution failed",
                extra={"tool": tool_name, "error": str(e)},
            )
            status = "failed_closed"
            result = {"error": str(e)}

        latency_ms = int((time.monotonic() - start_time) * 1000)
        traj = self._make_trajectory_entry(
            node="candidate_agent",
            tool_used=tool_name,
            latency_ms=latency_ms,
            model=None,  # model tracked by router; not repeated here to avoid duplication
            status=status,
            error_detail=error_detail if status != "success" else None,
        )
        return result, traj

    def _make_trajectory_entry(
        self,
        node: str,
        tool_used: Optional[str],
        latency_ms: int,
        model: Optional[str],
        status: str,
        *,
        input_data: Any = None,
        output_data: Any = None,
        error_detail: Optional[str] = None,
    ) -> TrajectoryEntry:
        """Build a TrajectoryEntry for audit logging (ADR-8).

        Computes real sha256 hashes of input/output data when provided.
        Falls back to '0' * 64 placeholder when data is not available
        (e.g. for interrupt-based tools like ask_candidate).

        Args:
            node: The node name (e.g. "candidate_agent").
            tool_used: The tool name, or None for fixed-node entries.
            latency_ms: Execution latency in milliseconds.
            model: The model identifier string.
            status: Execution status ("success", "retried", "failed_closed",
                    "failed_closed_quota_exhausted", "escalated").
            input_data: The input payload to hash (optional).
            output_data: The output payload to hash (optional).
            error_detail: Human-readable error cause for non-success entries.

        Returns:
            A fully populated TrajectoryEntry.
        """
        from scoutai.audit.log import compute_sha256
        now = datetime.now(timezone.utc).isoformat()

        input_hash = compute_sha256(input_data) if input_data is not None else "0" * 64
        output_hash = compute_sha256(output_data) if output_data is not None else "0" * 64

        return TrajectoryEntry(
            node=node,
            tool_used=tool_used,
            timestamp=now,
            input_hash=input_hash,
            output_hash=output_hash,
            latency_ms=max(0, latency_ms),
            model=model,
            model_role="fast_structured",
            prompt_version=self._config.prompt_versions.get(tool_used or "extract_evidence", "1.0.0"),
            tool_version=self._config.tool_versions.get(tool_used or "extract_evidence", "1.0.0"),
            schema_version=self._schema_version,
            status=status,  # type: ignore[arg-type]
            error_detail=error_detail,
        )

    def _build_task_message(
        self, candidate: CandidateState, role_profile: RoleProfile, rubric: Rubric
    ) -> str:
        """Build the initial task message with candidate context."""
        rubric_summary = ", ".join(
            f"{c.name} (weight={c.weight})" for c in rubric.criteria[:5]
        )
        return (
            f"Evaluate this candidate for the role: {role_profile.title}\n\n"
            f"Rubric criteria: {rubric_summary}{'...' if len(rubric.criteria) > 5 else ''}\n\n"
            f"Candidate ID: {candidate.candidate_id}\n"
            f"Résumé length: {len(candidate.sanitized_resume)} characters\n\n"
            f"Begin by calling extract_evidence to gather evidence from the résumé."
        )

    def _build_tools_description(self, available_tools: set[str]) -> str:
        return ", ".join(sorted(available_tools))

    def _get_langchain_tools(
        self,
        available_tools: set[str],
        candidate: CandidateState,
        role_profile: RoleProfile,
        rubric: Rubric,
    ) -> list[Any]:
        """
        Build LangChain tool definitions for the available tools.

        These are passed to model.bind_tools() so the LLM knows the tool signatures.
        """
        from langchain_core.tools import tool as lc_tool

        tools = []

        if "extract_evidence" in available_tools:
            @lc_tool("extract_evidence")
            def extract_evidence_tool() -> str:
                """Extract evidence from the candidate's résumé relevant to the role requirements."""
                return "call extract_evidence"
            tools.append(extract_evidence_tool)

        if "assess_capabilities" in available_tools:
            @lc_tool("assess_capabilities")
            def assess_capabilities_tool() -> str:
                """Assess the candidate's capabilities against the rubric using extracted evidence."""
                return "call assess_capabilities"
            tools.append(assess_capabilities_tool)

        if "verify_evidence" in available_tools:
            @lc_tool("verify_evidence")
            def verify_evidence_tool() -> str:
                """Verify whether the evidence is sufficient to make a recommendation."""
                return "call verify_evidence"
            tools.append(verify_evidence_tool)

        if "generate_interview_questions" in available_tools:
            @lc_tool("generate_interview_questions")
            def generate_interview_questions_tool() -> str:
                """Generate interview questions to address evidence gaps. Call at most once."""
                return "call generate_interview_questions"
            tools.append(generate_interview_questions_tool)

        if "ask_candidate" in available_tools:
            @lc_tool("ask_candidate")
            def ask_candidate_tool(question: str, target_criterion: str, rationale: str, priority_score: float = 1.0) -> str:
                """Ask the candidate one clarifying question. This tool can only be used once."""
                return f"interrupt: ask_candidate about {target_criterion}"
            tools.append(ask_candidate_tool)

        if "reevaluate_candidate" in available_tools:
            @lc_tool("reevaluate_candidate")
            def reevaluate_candidate_tool(question: str, answer: str) -> str:
                """Update the candidate's scores based on their interview answer."""
                return "call reevaluate_candidate"
            tools.append(reevaluate_candidate_tool)

        if "finalize_candidate" in available_tools:
            @lc_tool("finalize_candidate")
            def finalize_candidate_tool(
                recommendation: str,
                rationale: str,
                remaining_uncertainties: list[str] = [],
            ) -> str:
                """REQUIRED: Finalize the evaluation with a recommendation. Options: reject|hold|interview|strong_interview"""
                return f"finalize: {recommendation}"
            tools.append(finalize_candidate_tool)

        return tools

    def _serialize_tool_result(self, tool_name: str, result: Any) -> str:
        """Serialize a tool result to a string for the message history."""
        if result is None:
            return "null"
        if isinstance(result, dict):
            return json.dumps(result, default=str)[:2000]
        if hasattr(result, "model_dump"):
            return result.model_dump_json()[:2000]
        return str(result)[:2000]
