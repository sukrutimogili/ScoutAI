"""
LangGraph outer graph nodes — deterministic, fixed-node capabilities.

Each node has exactly one responsibility and wraps one capability.
The agent has NO influence over these nodes (§3.2, ADR-9).

Nodes implemented here:
- setup_role         → calls extract_role_requirements
- build_rubric       → calls generate_rubric
- select_candidate   → picks next unprocessed candidate, or routes to fairness_probe
- screen_resume_node → calls screen_resume capability
- candidate_agent_node → runs AgentHarness for one candidate
- fairness_probe_node  → calls run_fairness_probe (S9, stub here)
- compose_summary_node → calls compose_decision_summary (S9, stub here)
- human_review_node    → LangGraph interrupt (S11, stub here)
- schedule_node        → calls check_availability + propose_interview (S10, stub here)

Routing functions (conditional edges):
- route_after_select  → screen_resume | fairness_probe
- route_after_screen  → human_review | candidate_agent
- route_after_human_review → schedule | select_candidate | END

Spec reference: §3.2 (LangGraph node/edge structure).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from scoutai.agent.harness import AgentHarness
from scoutai.capabilities.fairness import compose_decision_summary, run_fairness_probe_on_shortlist
from scoutai.capabilities.model_router import ModelRouter
from scoutai.capabilities.scheduling import check_availability, propose_interview
from scoutai.capabilities.role_requirements import SessionCache, extract_role_requirements, generate_rubric
from scoutai.capabilities.screen_resume import apply_screen_result_to_candidate, screen_resume
from scoutai.config import ScoutAIConfig
from scoutai.schemas import CandidateState, GraphState, RecruiterSummary, ShortlistEntry

logger = logging.getLogger(__name__)


# ── Node: setup_role ──────────────────────────────────────────────────────────


def setup_role_node(state: dict[str, Any], config: ScoutAIConfig, router: ModelRouter, cache: SessionCache) -> dict[str, Any]:
    """
    Extract structured role requirements from the job description.

    Fixed node — runs once per session (cached by jd_hash).
    """
    jd = state.get("jd", "")
    logger.info("setup_role: extracting role requirements")
    role_profile = extract_role_requirements(jd, config, router, cache=cache)
    return {
        "role_profile": role_profile,
        "step_count": state.get("step_count", 0) + 1,
    }


# ── Node: build_rubric ────────────────────────────────────────────────────────


def build_rubric_node(state: dict[str, Any], config: ScoutAIConfig, router: ModelRouter, cache: SessionCache) -> dict[str, Any]:
    """
    Generate evaluation rubric from the role profile.

    Fixed node — runs once per session (cached by jd_hash + rubric_config_version).
    """
    role_profile = state.get("role_profile")
    if role_profile is None:
        raise ValueError("build_rubric: role_profile not set — setup_role must run first")
    logger.info("build_rubric: generating rubric")
    rubric = generate_rubric(role_profile, config, router, cache=cache)
    return {
        "rubric": rubric,
        "step_count": state.get("step_count", 0) + 1,
    }


# ── Node: select_candidate ────────────────────────────────────────────────────


def select_candidate_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Pick the next unprocessed candidate, or signal completion.

    This is the loop-control node. It sets current_idx to the first candidate
    whose finalized=False. When all candidates are finalized, it sets current_idx
    to -1 to signal routing to fairness_probe.

    Stopping condition (§3.3): routes to fairness_probe only when EVERY candidate
    has finalized=True — explicit 'done' state, never implicit fall-through.
    """
    candidates = state.get("candidates", [])

    # Find next unfinalized candidate
    for idx, candidate in enumerate(candidates):
        finalized = candidate.get("finalized", False) if isinstance(candidate, dict) else getattr(candidate, "finalized", False)
        if not finalized:
            logger.info(
                "select_candidate: selected candidate",
                extra={"index": idx, "total": len(candidates)},
            )
            return {
                "current_idx": idx,
                "step_count": state.get("step_count", 0) + 1,
            }

    # All candidates finalized — signal to proceed to fairness_probe
    logger.info("select_candidate: all candidates finalized — routing to fairness_probe")
    return {
        "current_idx": -1,  # sentinel: all done
        "step_count": state.get("step_count", 0) + 1,
    }


def route_after_select(state: dict[str, Any]) -> Literal["screen_resume", "fairness_probe"]:
    """
    Conditional edge after select_candidate.

    - current_idx == -1 → all finalized → fairness_probe
    - current_idx >= 0  → more candidates → screen_resume
    """
    current_idx = state.get("current_idx", 0)
    if current_idx == -1:
        return "fairness_probe"
    return "screen_resume"


# ── Node: screen_resume_node ──────────────────────────────────────────────────


def screen_resume_node(state: dict[str, Any], config: ScoutAIConfig, router: ModelRouter) -> dict[str, Any]:
    """
    Screen the current candidate's résumé for PII and injection.

    Fixed node — runs BEFORE candidate_agent ever sees the résumé (ADR-9).
    Updates the candidate in the candidates list.
    """
    candidates = list(state.get("candidates", []))
    current_idx = state.get("current_idx", 0)

    if current_idx < 0 or current_idx >= len(candidates):
        raise ValueError(f"screen_resume: invalid current_idx={current_idx}")

    raw = candidates[current_idx]
    candidate: CandidateState = raw if isinstance(raw, CandidateState) else CandidateState.model_validate(raw)

    logger.info("screen_resume: screening candidate", extra={"candidate_id": candidate.candidate_id, "idx": current_idx})

    result = screen_resume(
        resume_text=candidate.resume_text,
        config=config,
        router=router,
        candidate_id=candidate.candidate_id,
    )

    updated = apply_screen_result_to_candidate(candidate, result)
    candidates[current_idx] = updated

    return {
        "candidates": candidates,
        "step_count": state.get("step_count", 0) + 1,
    }


def route_after_screen(state: dict[str, Any]) -> Literal["human_review", "candidate_agent"]:
    """
    Conditional edge after screen_resume.

    - injection_flag OR leakage_flag → human_review (flagged résumé must NOT reach agent)
    - clean → candidate_agent (normal path)
    """
    candidates = state.get("candidates", [])
    current_idx = state.get("current_idx", 0)

    if current_idx < 0 or current_idx >= len(candidates):
        return "human_review"

    candidate = candidates[current_idx]
    injection = candidate.get("injection_flag", False) if isinstance(candidate, dict) else getattr(candidate, "injection_flag", False)
    leakage = candidate.get("leakage_flag", False) if isinstance(candidate, dict) else getattr(candidate, "leakage_flag", False)

    if injection or leakage:
        logger.warning(
            "route_after_screen: flagged résumé → human_review",
            extra={"injection_flag": injection, "leakage_flag": leakage},
        )
        return "human_review"

    return "candidate_agent"


# ── Node: candidate_agent_node ────────────────────────────────────────────────


def candidate_agent_node(state: dict[str, Any], config: ScoutAIConfig, router: ModelRouter) -> dict[str, Any]:
    """
    Run the bounded ReAct agent loop for the current candidate (ADR-9).

    After the agent finalizes, the candidate's state and the ShortlistEntry are
    written back into graph state, then select_candidate is called again.
    """
    candidates = list(state.get("candidates", []))
    current_idx = state.get("current_idx", 0)
    role_profile = state.get("role_profile")
    rubric = state.get("rubric")

    if role_profile is None or rubric is None:
        raise ValueError("candidate_agent: role_profile and rubric must be set")

    candidate: CandidateState = (
        candidates[current_idx]
        if isinstance(candidates[current_idx], CandidateState)
        else CandidateState.model_validate(candidates[current_idx])
    )

    logger.info(
        "candidate_agent: starting evaluation",
        extra={"candidate_id": candidate.candidate_id, "idx": current_idx},
    )

    harness = AgentHarness(config, router)
    updated_candidate, shortlist_entry, new_traj = harness.run(candidate, role_profile, rubric)

    # Write back updated candidate
    candidates[current_idx] = updated_candidate

    # Append to shortlist if we have a valid entry
    shortlist = list(state.get("shortlist", []))
    if shortlist_entry is not None:
        shortlist.append(shortlist_entry)

    # Append new trajectory entries (ADR-8)
    trajectory = list(state.get("trajectory", []))
    trajectory.extend(new_traj)

    logger.info(
        "candidate_agent: evaluation complete",
        extra={
            "candidate_id": candidate.candidate_id,
            "recommendation": updated_candidate.recommendation,
        },
    )

    return {
        "candidates": candidates,
        "shortlist": shortlist,
        "trajectory": trajectory,
        "step_count": state.get("step_count", 0) + 1,
    }


# ── Node stubs for S9–S11 ─────────────────────────────────────────────────────
# These are implemented fully in their respective steps.
# Stubs here allow the graph to be assembled and tested end-to-end in S8.


def fairness_probe_node(state: dict[str, Any], config: ScoutAIConfig, router: ModelRouter) -> dict[str, Any]:
    """
    Run fairness probe on shortlisted candidates (ADR-7, S9).

    Operates on the shortlist only — the expensive counterfactual bias check is
    reserved for candidates that are being seriously considered (ADR-7).
    Runs pairwise comparisons across all adjacent shortlist pairs.
    """
    shortlist_raw = state.get("shortlist", [])
    shortlist = [
        e if isinstance(e, ShortlistEntry) else ShortlistEntry.model_validate(e)
        for e in shortlist_raw
    ]

    logger.info(
        "fairness_probe: running probes on shortlist",
        extra={"shortlist_count": len(shortlist)},
    )

    bias_reports = run_fairness_probe_on_shortlist(shortlist, config, router)

    logger.info(
        "fairness_probe: completed",
        extra={"reports_generated": len(bias_reports)},
    )

    return {
        "bias_reports": bias_reports,
        "step_count": state.get("step_count", 0) + 1,
    }


def compose_summary_node(state: dict[str, Any], config: ScoutAIConfig, router: ModelRouter) -> dict[str, Any]:
    """
    Compose recruiter decision summary (§4.1, S9).

    Uses the high_context model role (1x per session).
    Every claim in the summary must cite evidence_refs — enforced by the
    RecruiterSummary schema and the compose_decision_summary capability (§4.1).
    """
    shortlist_raw = state.get("shortlist", [])
    shortlist = [
        e if isinstance(e, ShortlistEntry) else ShortlistEntry.model_validate(e)
        for e in shortlist_raw
    ]

    bias_reports_raw = state.get("bias_reports", [])
    from scoutai.schemas import BiasReport
    bias_reports = [
        r if isinstance(r, BiasReport) else BiasReport.model_validate(r)
        for r in bias_reports_raw
    ]

    run_id = state.get("run_id", "")

    logger.info(
        "compose_summary: generating recruiter summary",
        extra={"shortlist_count": len(shortlist), "bias_reports": len(bias_reports)},
    )

    summary = compose_decision_summary(
        shortlist=shortlist,
        bias_reports=bias_reports,
        config=config,
        router=router,
        run_id=run_id,
    )

    logger.info(
        "compose_summary: completed",
        extra={"evidence_refs_count": len(summary.evidence_refs)},
    )

    return {
        "recruiter_summary": summary,
        "step_count": state.get("step_count", 0) + 1,
    }


def human_review_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Human-in-the-loop interrupt node (§3.2, S11).

    This node uses LangGraph's interrupt mechanism to pause execution and
    wait for a human recruiter decision. The caller injects the decision
    via the checkpointer's `Command` or by resuming the thread with input.

    The node supports four actions (§3.2):
    - approve:        Accept the recommendation as-is. Routes to schedule node.
    - reject:         Reject the recommendation. Routes to END.
    - edit_recommendation: Change the recommendation (e.g. interview→hold).
                       Routes to END after applying the edit.
    - request-more-evidence: Send a specific candidate back through the
                       candidate_agent for another evaluation pass.
                       The candidate's interview_rounds is reset for the
                       reopened cycle only, and the reopen is logged as a
                       human override (ADR-3).

    The human_review_decision is written to state so the routing function
    can determine the next node. The decision payload includes:
    {
        "action": "approve" | "reject" | "edit_recommendation" | "request-more-evidence",
        "candidate_id": Optional[str],      # required for request-more-evidence
        "new_recommendation": Optional[str], # required for edit_recommendation
        "rationale": Optional[str],          # human's reasoning
        "overridden_at": str                 # ISO 8601 timestamp
    }

    Spec references: §3.2, ADR-3.
    """
    from langgraph.types import interrupt

    shortlist_raw = state.get("shortlist", [])
    shortlist = [
        e if isinstance(e, ShortlistEntry) else ShortlistEntry.model_validate(e)
        for e in shortlist_raw
    ]

    recruiter_summary = state.get("recruiter_summary")
    bias_reports = state.get("bias_reports", [])

    # Build the review payload presented to the human
    review_payload = {
        "shortlist": [
            {
                "candidate": e.candidate,
                "recommendation": e.recommendation,
                "weighted_score": e.weighted_score,
                "strengths": e.strengths,
                "remaining_uncertainties": e.remaining_uncertainties,
                "evidence_refs_count": len(e.evidence_refs),
            }
            for e in shortlist
        ],
        "bias_reports_count": len(bias_reports),
        "overall_recommendation": (
            recruiter_summary.overall_recommendation
            if hasattr(recruiter_summary, "overall_recommendation")
            else None
        ),
        "message": (
            "Review the shortlist above. Provide your decision: "
            "approve, reject, edit_recommendation, or request-more-evidence."
        ),
    }

    logger.info(
        "human_review: waiting for human decision",
        extra={
            "shortlist_count": len(shortlist),
            "bias_reports_count": len(bias_reports),
        },
    )

    # Interrupt — execution pauses here. The caller resumes with a decision.
    decision = interrupt(review_payload)

    # Validate the decision structure
    action = decision.get("action", "reject") if isinstance(decision, dict) else "reject"
    candidate_id = decision.get("candidate_id") if isinstance(decision, dict) else None
    new_recommendation = decision.get("new_recommendation") if isinstance(decision, dict) else None
    rationale = decision.get("rationale") if isinstance(decision, dict) else None

    valid_actions = {"approve", "reject", "edit_recommendation", "request-more-evidence"}
    if action not in valid_actions:
        logger.warning(
            "human_review: invalid action, defaulting to 'reject'",
            extra={"received_action": action},
        )
        action = "reject"

    human_decision = {
        "action": action,
        "candidate_id": candidate_id,
        "new_recommendation": new_recommendation,
        "rationale": rationale,
        "overridden_at": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "human_review: decision received",
        extra={
            "action": action,
            "candidate_id": candidate_id,
            "new_recommendation": new_recommendation,
        },
    )

    # Apply action outcomes
    result_updates: dict[str, Any] = {
        "human_review_decision": human_decision,
        "step_count": state.get("step_count", 0) + 1,
    }

    if action == "edit_recommendation" and new_recommendation and shortlist:
        # Update the shortlist entry with the new recommendation
        updated_shortlist = list(shortlist)
        if candidate_id:
            # Update a specific candidate
            found = False
            for i, entry in enumerate(updated_shortlist):
                if entry.candidate == candidate_id:
                    updated_shortlist[i] = ShortlistEntry(
                        candidate=entry.candidate,
                        recommendation=new_recommendation,  # type: ignore[arg-type]
                        weighted_score=entry.weighted_score,
                        confidence_summary=entry.confidence_summary,
                        strengths=entry.strengths,
                        remaining_uncertainties=entry.remaining_uncertainties,
                        evidence_refs=entry.evidence_refs,
                    )
                    found = True
                    break
            if not found:
                # Candidate not found — update the last entry as fallback
                last = updated_shortlist[-1]
                updated_shortlist[-1] = ShortlistEntry(
                    candidate=last.candidate,
                    recommendation=new_recommendation,  # type: ignore[arg-type]
                    weighted_score=last.weighted_score,
                    confidence_summary=last.confidence_summary,
                    strengths=last.strengths,
                    remaining_uncertainties=last.remaining_uncertainties,
                    evidence_refs=last.evidence_refs,
                )
        else:
            # No candidate_id specified — update the last entry
            last = updated_shortlist[-1]
            updated_shortlist[-1] = ShortlistEntry(
                candidate=last.candidate,
                recommendation=new_recommendation,  # type: ignore[arg-type]
                weighted_score=last.weighted_score,
                confidence_summary=last.confidence_summary,
                strengths=last.strengths,
                remaining_uncertainties=last.remaining_uncertainties,
                evidence_refs=last.evidence_refs,
            )
        result_updates["shortlist"] = updated_shortlist

    if action == "request-more-evidence":
        # Reset the targeted candidate's finalized flag and interview_rounds (ADR-3)
        candidates = list(state.get("candidates", []))
        updated_candidates = list(candidates)
        for i, cand in enumerate(candidates):
            cid = cand.get("candidate_id") if isinstance(cand, dict) else getattr(cand, "candidate_id", "")
            if cid == candidate_id or (candidate_id is None and i == (state.get("current_idx", 0))):
                if isinstance(cand, dict):
                    updated_candidates[i] = {
                        **cand,
                        "finalized": False,
                        "interview_rounds": 0,  # Reset for reopened cycle (ADR-3)
                    }
                else:
                    updated_candidates[i] = cand.model_copy(
                        update={"finalized": False, "interview_rounds": 0}
                    )
                logger.info(
                    "human_review: candidate reset for re-evaluation",
                    extra={"candidate_id": cid, "idx": i},
                )
                break
        result_updates["candidates"] = updated_candidates

    return result_updates


def route_after_human_review(state: dict[str, Any]) -> Literal["schedule", "select_candidate", "END"]:
    """
    Conditional edge after human_review (§3.2, S11).

    Routing rules:
    - approve  → schedule (if recommendation allows interview/strong_interview)
    - approve  → END (if recommendation is reject/hold)
    - reject   → END
    - edit_recommendation → END (changes applied directly in the node)
    - request-more-evidence → select_candidate (re-opens candidate for re-evaluation)

    When routing to select_candidate (request-more-evidence), the targeted
    candidate's 'finalized' flag and 'interview_rounds' are reset. This is
    logged as a human override (ADR-3).
    """
    decision = state.get("human_review_decision")
    if not isinstance(decision, dict):
        logger.warning("route_after_human_review: no decision found, defaulting to END")
        return "END"

    action = decision.get("action", "reject")

    if action == "approve":
        # Check if any shortlist entry has a schedulable recommendation
        shortlist_raw = state.get("shortlist", [])
        schedulable = {"interview", "strong_interview"}
        for entry in shortlist_raw:
            rec = entry.get("recommendation") if isinstance(entry, dict) else getattr(entry, "recommendation", "")
            if rec in schedulable:
                logger.info("route_after_human_review: approved with interview recommendation → schedule")
                return "schedule"
        logger.info("route_after_human_review: approved without interview recommendation → END")
        return "END"

    if action == "request-more-evidence":
        candidate_id = decision.get("candidate_id")
        logger.info(
            "route_after_human_review: requesting more evidence for candidate",
            extra={"candidate_id": candidate_id},
        )
        # Candidate reset is handled in human_review_node. Route to select_candidate
        # so the per-candidate loop finds the re-opened candidate.
        return "select_candidate"

    # reject, edit_recommendation, or any other action → END
    logger.info("route_after_human_review: no further processing → END")
    return "END"


def schedule_node(state: dict[str, Any], config: ScoutAIConfig) -> dict[str, Any]:
    """
    Schedule interviews for approved candidates (S10).

    For each candidate in the shortlist whose recommendation allows scheduling
    (interview or strong_interview), check availability and propose an interview
    slot.

    Fixed node — deterministic (calendar API), not LLM-driven.
    NOT callable by the agent (§7.4, ADR-9).

    Uses the mock calendar backend by default (§12 Open Items).
    Google Calendar backend is the documented upgrade path.

    Args:
        state:  Graph state with shortlist and candidates.
        config: ScoutAI configuration.

    Returns:
        Updated state with slot information written to candidates.
    """
    candidates_raw = state.get("candidates", [])
    shortlist_raw = state.get("shortlist", [])

    # Parse shortlist entries
    shortlist = [
        e if isinstance(e, ShortlistEntry) else ShortlistEntry.model_validate(e)
        for e in shortlist_raw
    ]

    # Week start: next Monday from current time
    from datetime import datetime as dt_mod, timezone as tz_mod, timedelta as td_mod
    today = dt_mod.now(tz_mod.utc)
    days_until_monday = (7 - today.weekday()) % 7 or 7
    week_start = (today + td_mod(days=days_until_monday)).strftime("%Y-%m-%d")

    schedulable_recommendations = {"interview", "strong_interview"}
    updated_candidates = list(candidates_raw)

    for entry in shortlist:
        if entry.recommendation not in schedulable_recommendations:
            logger.info(
                "schedule: skipping candidate (non-interview recommendation)",
                extra={"candidate": entry.candidate, "recommendation": entry.recommendation},
            )
            continue

        try:
            slots = check_availability(
                candidate_id=entry.candidate,
                week_start=week_start,
                config=config,
            )
        except Exception as e:
            logger.error(
                "schedule: check_availability failed",
                extra={"candidate": entry.candidate, "error": str(e)},
            )
            continue

        if not slots:
            logger.warning(
                "schedule: no available slots for candidate",
                extra={"candidate": entry.candidate, "week_start": week_start},
            )
            continue

        try:
            confirmation = propose_interview(
                candidate_id=entry.candidate,
                slot=slots[0],  # Use the first available slot
                config=config,
            )
        except Exception as e:
            logger.error(
                "schedule: propose_interview failed",
                extra={"candidate": entry.candidate, "slot": slots[0].start_iso, "error": str(e)},
            )
            continue

        # Write slot info back to the corresponding candidate
        for i, cand in enumerate(updated_candidates):
            cid = cand.get("candidate_id") if isinstance(cand, dict) else getattr(cand, "candidate_id", "")
            if cid == entry.candidate:
                slot_info = {
                    "start_iso": slots[0].start_iso,
                    "end_iso": slots[0].end_iso,
                    "confirmation_id": confirmation.confirmation_id,
                    "status": confirmation.status,
                }
                if isinstance(updated_candidates[i], dict):
                    updated_candidates[i] = {**updated_candidates[i], "slot": slot_info}
                else:
                    updated_candidates[i] = updated_candidates[i].model_copy(
                        update={"slot": slot_info}
                    )
                break

        logger.info(
            "schedule: interview proposed",
            extra={
                "candidate": entry.candidate,
                "slot_start": slots[0].start_iso,
                "confirmation_id": confirmation.confirmation_id,
            },
        )

    return {
        "candidates": updated_candidates,
        "step_count": state.get("step_count", 0) + 1,
    }
