"""
Hiring Run Summary — the one screen a recruiter should understand in 3 seconds.

Big display number (total candidates), recommendation distribution, fairness alerts,
review progress, and two actions.

Uses real pipeline data from session state (set by processing.py on completion).
"""

from __future__ import annotations

import streamlit as st

from ui.components import masthead, status_pill, stat_block, progress_bar, alert_warn


def _get_candidates() -> list[dict]:
    """Get candidate data from pipeline result or fall back to empty list."""
    data = st.session_state.get("pipeline_data", {})
    candidates_raw = data.get("candidates", [])
    result = []
    for c in candidates_raw:
        if isinstance(c, dict):
            result.append(c)
        elif hasattr(c, "model_dump"):
            result.append(c.model_dump())
        else:
            result.append(dict(c))
    return result


def _get_bias_reports() -> list[dict]:
    """Get bias reports from pipeline result."""
    data = st.session_state.get("pipeline_data", {})
    reports = data.get("bias_reports", [])
    result = []
    for r in reports:
        if isinstance(r, dict):
            result.append(r)
        elif hasattr(r, "model_dump"):
            result.append(r.model_dump())
        else:
            result.append(dict(r))
    return result


def render() -> None:
    """Render the hiring run summary screen."""
    run_name = st.session_state.get("run_name", "Hiring Run")
    masthead(run_name)

    st.markdown('<div class="content">', unsafe_allow_html=True)

    candidates = _get_candidates()
    bias_reports = _get_bias_reports()

    # Count recommendations
    total = len(candidates)
    strong = sum(1 for c in candidates if c.get("recommendation") == "strong_interview")
    interview = sum(1 for c in candidates if c.get("recommendation") == "interview")
    reject = sum(1 for c in candidates if c.get("recommendation") == "reject")
    needs_review = sum(1 for c in candidates if c.get("recommendation") == "hold")

    # Big display number (magazine "pull quote")
    st.markdown(
        f'<p class="display-number">{total}</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="caption" style="margin-top:4px;margin-bottom:32px;">Total candidates</p>',
        unsafe_allow_html=True,
    )

    # Recommendation distribution — four stat blocks
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(stat_block(strong, "Strong Shortlist"), unsafe_allow_html=True)
    with col2:
        st.markdown(stat_block(interview, "Interview"), unsafe_allow_html=True)
    with col3:
        st.markdown(stat_block(reject, "Reject"), unsafe_allow_html=True)
    with col4:
        st.markdown(stat_block(needs_review, "Needs Review"), unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Fairness alerts — only rendered if count > 0
    if bias_reports:
        st.markdown(
            alert_warn(
                f"Fairness alerts: {len(bias_reports)} report(s) requiring attention."
            ),
            unsafe_allow_html=True,
        )

    # Candidates requiring validation
    pending = [c for c in candidates if c.get("remaining_uncertainties")]
    if pending:
        st.markdown(
            alert_warn(
                f"Candidates requiring validation: {len(pending)} candidate(s) with unresolved uncertainties."
            ),
            unsafe_allow_html=True,
        )

    # Review progress
    reviewed = sum(1 for c in candidates if c.get("finalized", False))
    st.markdown(
        f'<p class="caption" style="margin-bottom:8px;">{reviewed} of {total} reviewed</p>',
        unsafe_allow_html=True,
    )
    st.markdown(progress_bar(reviewed, total), unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Actions row
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Review candidates", type="primary", use_container_width=True, key="summary_review"):
            st.session_state["screen"] = "candidate_list"
            st.rerun()
    with col2:
        if st.button("Export report", type="secondary", use_container_width=True, key="summary_export"):
            st.session_state["screen"] = "complete"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)