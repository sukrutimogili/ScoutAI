"""
Candidate Review — the deepest screen, progressively disclosed.

Rendered in exact vertical order: Recommendation → Why? → Needs Validation →
Interview Focus → Supporting Evidence → Similar Candidates → Actions.
"""

from __future__ import annotations

import streamlit as st

from ui.components import (
    masthead,
    status_pill,
    evidence_list,
    expandable_section,
    alert_warn,
)
from ui.mock_data import MOCK_CANDIDATES, MOCK_SHORTLIST


def render() -> None:
    """Render the candidate review screen."""
    run_name = st.session_state.get("run_name", "Hiring Run")
    masthead(run_name)

    candidate_id = st.session_state.get("current_candidate_id", "c001")
    candidate = next(
        (c for c in MOCK_CANDIDATES if c["candidate_id"] == candidate_id),
        MOCK_CANDIDATES[0],
    )

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

    # Generate interview focus areas from uncertainties and low-confidence capabilities
    capabilities = candidate.get("capabilities", {})
    low_conf = [
        name for name, cap in capabilities.items()
        if cap.get("confidence", "high") in ("low", "unknown")
    ]
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

    evidence = candidate.get("evidence_buckets", {})
    for category, items in evidence.items():
        st.markdown(
            f'<p style="font-size:13px;font-weight:700;color:var(--muted);margin-top:16px;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.05em;">{category}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(evidence_list(items), unsafe_allow_html=True)

    # 6. Similar Candidates
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2>Similar Candidates</h2>", unsafe_allow_html=True)

    other_candidates = [c for c in MOCK_CANDIDATES if c["candidate_id"] != candidate_id]
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
            # In production, this would call resume_with_decision
            st.session_state["screen"] = "continue_reviewing"
            st.session_state["last_action"] = "approved"
            st.rerun()
    with col2:
        if st.button("Reject", type="secondary", use_container_width=True, key="review_reject"):
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