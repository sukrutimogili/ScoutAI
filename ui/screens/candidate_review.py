"""
Candidate Review — the deepest screen, progressively disclosed.

Rendered in exact vertical order: Recommendation → Why? → Needs Validation →
Interview Focus → Supporting Evidence → Similar Candidates → Actions.
"""

from __future__ import annotations

import logging

import streamlit as st

from ui.components import (
    masthead,
    status_pill,
    evidence_list,
    expandable_section,
    alert_warn,
)
from ui.mock_data import MOCK_CANDIDATES, MOCK_SHORTLIST

logger = logging.getLogger(__name__)


def _get_candidates() -> list[dict]:
    """Return candidates from real pipeline_data, falling back to mock data."""
    data = st.session_state.get("pipeline_data", {})
    candidates_raw = data.get("candidates", [])
    if candidates_raw:
        return [c if isinstance(c, dict) else c.model_dump() for c in candidates_raw]
    return list(MOCK_CANDIDATES)


def _get_candidate(candidate_id: str) -> dict:
    """Fetch a single candidate by ID from pipeline_data, falling back to mock."""
    for c in _get_candidates():
        if c.get("candidate_id") == candidate_id:
            return c
    # Fallback: first mock candidate
    return next(
        (c for c in MOCK_CANDIDATES if c["candidate_id"] == candidate_id),
        MOCK_CANDIDATES[0],
    )


def _render_evidence(candidate: dict) -> None:
    """
    Render the Supporting Evidence section.

    Handles two shapes:
      - Real backend: evidence_buckets is list[dict] with keys value/source/jd_relevance
        (serialised EvidenceItem objects from CandidateState)
      - Mock data: evidence_buckets is dict[str, list[str]]
    """
    evidence_raw = candidate.get("evidence_buckets", [])

    if isinstance(evidence_raw, dict):
        # Mock data shape: {category: [str, ...]}
        for category, items in evidence_raw.items():
            st.markdown(
                f'<p style="font-size:13px;font-weight:700;color:var(--muted);'
                f'margin-top:16px;margin-bottom:8px;text-transform:uppercase;'
                f'letter-spacing:0.05em;">{category}</p>',
                unsafe_allow_html=True,
            )
            st.markdown(evidence_list(items), unsafe_allow_html=True)

    elif isinstance(evidence_raw, list) and evidence_raw:
        # Real backend shape: [{value, source, jd_relevance}, ...]
        # Group items by source for cleaner display
        by_source: dict[str, list[str]] = {}
        for item in evidence_raw:
            if not isinstance(item, dict):
                continue
            source = item.get("source", "General")
            value = item.get("value", "")
            if value:
                by_source.setdefault(source, []).append(value)

        if by_source:
            for source, values in by_source.items():
                st.markdown(
                    f'<p style="font-size:13px;font-weight:700;color:var(--muted);'
                    f'margin-top:16px;margin-bottom:8px;text-transform:uppercase;'
                    f'letter-spacing:0.05em;">{source}</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(evidence_list(values), unsafe_allow_html=True)
        else:
            st.markdown(
                '<p class="caption">No supporting evidence available.</p>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<p class="caption">No supporting evidence available.</p>',
            unsafe_allow_html=True,
        )


def _render_capabilities_for_interview_focus(candidate: dict) -> list[str]:
    """
    Return a list of low-confidence capability names.

    Handles two shapes for capabilities values:
      - Real backend: CapabilityAssessment dict with confidence/evidence_refs
      - Mock data: dict with score/confidence
    """
    capabilities = candidate.get("capabilities", {})
    low_conf = []
    for name, cap in capabilities.items():
        if not isinstance(cap, dict):
            continue
        confidence = cap.get("confidence", "high")
        if confidence in ("low", "unknown"):
            low_conf.append(name)
    return low_conf


def render() -> None:
    """Render the candidate review screen."""
    run_name = st.session_state.get("run_name", "Hiring Run")
    masthead(run_name)

    candidate_id = st.session_state.get("current_candidate_id", "c001")
    candidate = _get_candidate(candidate_id)

    st.markdown('<div class="content">', unsafe_allow_html=True)

    # 1. Recommendation — status pill + name
    st.markdown(
        f"""
        <div style="margin-bottom:24px;">
            {status_pill(candidate["recommendation"])}
            <h1 style="margin-top:12px;margin-bottom:0;">{candidate["candidate_id"]}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # 2. Why? — collapsed by default, expandable
    why_content = f"""
    <p>{candidate.get("rationale", "No rationale provided.")}</p>
    """
    st.markdown(
        expandable_section("Why?", why_content, expanded=False),
        unsafe_allow_html=True,
    )

    # 3. Needs Validation — only if non-empty
    uncertainties = candidate.get("remaining_uncertainties", [])
    if uncertainties:
        st.markdown(
            '<h2 style="margin-top:24px;">Needs Validation</h2>',
            unsafe_allow_html=True,
        )
        for u in uncertainties:
            st.markdown(alert_warn(u), unsafe_allow_html=True)

    # 4. Interview Focus — suggested questions/areas
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2>Interview Focus</h2>", unsafe_allow_html=True)

    low_conf = _render_capabilities_for_interview_focus(candidate)
    focus_areas = uncertainties + low_conf
    if focus_areas:
        for area in focus_areas:
            st.markdown(
                f'<div class="evidence-item">{area}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<p class="caption">No specific areas identified for interview focus.</p>',
            unsafe_allow_html=True,
        )

    # 5. Supporting Evidence
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2>Supporting Evidence</h2>", unsafe_allow_html=True)
    _render_evidence(candidate)

    # 6. Similar Candidates
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2>Similar Candidates</h2>", unsafe_allow_html=True)

    all_candidates = _get_candidates()
    other_candidates = [c for c in all_candidates if c.get("candidate_id") != candidate_id]
    for other in other_candidates[:2]:  # Max 2 suggestions
        other_strengths = other.get("strengths", [])
        similarity = "; ".join(other_strengths[:2]) if other_strengths else "Similar profile"
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(
                f"""
                <div style="padding:8px 0;border-bottom:1px solid var(--line);">
                    <span style="font-weight:700;">{other["candidate_id"]}</span>
                    <span class="caption" style="margin-left:8px;">{similarity}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            if st.button(
                "Compare",
                key=f"compare_{other['candidate_id']}",
                type="secondary",
            ):
                st.session_state["compare_candidate_id"] = other["candidate_id"]
                st.session_state["screen"] = "candidate_comparison"
                st.rerun()

    # 7. Actions — bottom of page
    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Approve", type="primary", use_container_width=True, key="review_approve"):
            _submit_decision(candidate_id, "approve")
            st.session_state["screen"] = "continue_reviewing"
            st.session_state["last_action"] = "approved"
            st.rerun()
    with col2:
        if st.button("Reject", type="secondary", use_container_width=True, key="review_reject"):
            _submit_decision(candidate_id, "reject")
            st.session_state["screen"] = "continue_reviewing"
            st.session_state["last_action"] = "rejected"
            st.rerun()
    with col3:
        if st.button("Ask candidate", key="review_ask_candidate"):
            st.info("Interview question queued for candidate.")

    # Back to list
    if st.button("Back to list", type="secondary", key="back_to_list"):
        st.session_state["screen"] = "candidate_list"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def _submit_decision(candidate_id: str, action: str) -> None:
    """
    Call resume_with_decision on the active thread, if one exists.

    Falls back gracefully when no real pipeline run is active (e.g. mock-data
    demo mode), so the navigation still works without crashing.
    """
    thread_id = st.session_state.get("thread_id")
    if not thread_id:
        logger.debug(
            "No active thread_id in session state — skipping resume_with_decision "
            "(running in mock/demo mode)."
        )
        return

    try:
        from scoutai.runtime.session import resume_with_decision

        decision = {"action": action, "candidate_id": candidate_id}
        result = resume_with_decision(thread_id, decision)
        logger.info(
            "resume_with_decision completed",
            extra={"thread_id": thread_id, "action": action, "candidate_id": candidate_id},
        )
        # Refresh pipeline_data in session state with the updated result
        if result:
            st.session_state["pipeline_data"] = result
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "resume_with_decision failed: %s", exc,
            extra={"thread_id": thread_id, "action": action, "candidate_id": candidate_id},
        )
        st.warning(f"Could not submit decision to pipeline: {exc}")
