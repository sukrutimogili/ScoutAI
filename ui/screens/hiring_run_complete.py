"""
Hiring Run Complete — final summary with export options.

Mirrors the Hiring Run Summary layout, relabeled as final.
"""

from __future__ import annotations

import streamlit as st

from ui.components import masthead, status_pill, stat_block, progress_bar, alert_warn
from ui.mock_data import MOCK_CANDIDATES, MOCK_BIAS_REPORTS


def render() -> None:
    """Render the hiring run complete screen."""
    run_name = st.session_state.get("run_name", "Hiring Run")
    masthead(run_name)

    st.markdown('<div class="content">', unsafe_allow_html=True)

    # Use mock data
    candidates = MOCK_CANDIDATES
    bias_reports = MOCK_BIAS_REPORTS

    # Count recommendations
    total = len(candidates)
    strong = sum(1 for c in candidates if c["recommendation"] == "strong_interview")
    interview = sum(1 for c in candidates if c["recommendation"] == "interview")
    reject = sum(1 for c in candidates if c["recommendation"] == "reject")
    needs_review = sum(1 for c in candidates if c["recommendation"] == "hold")

    st.markdown('<p class="eyebrow">HIRING RUN COMPLETE</p>', unsafe_allow_html=True)

    # Big display number
    st.markdown(
        f'<p class="display-number">{total}</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="caption" style="margin-top:4px;margin-bottom:32px;">Total candidates</p>',
        unsafe_allow_html=True,
    )

    # Recommendation distribution
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

    # Fairness alerts
    if bias_reports:
        st.markdown(
            alert_warn(
                f"Fairness alerts: {len(bias_reports)} report(s) requiring attention."
            ),
            unsafe_allow_html=True,
        )

    # Review progress (complete)
    reviewed = total
    st.markdown(
        f'<p class="caption" style="margin-bottom:8px;">{reviewed} of {total} reviewed</p>',
        unsafe_allow_html=True,
    )
    st.markdown(progress_bar(reviewed, total), unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Export section — three equal-weight outlined buttons
    st.markdown("<h2>Export</h2>", unsafe_allow_html=True)
    st.markdown(
        '<p class="caption" style="margin-bottom:16px;">Download the full report in your preferred format.</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("PDF", type="secondary", use_container_width=True, key="export_pdf")
    with col2:
        st.button("CSV", type="secondary", use_container_width=True, key="export_csv")
    with col3:
        st.button("JSON", type="secondary", use_container_width=True, key="export_json")

    # Start new run
    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button(
        "Start new hiring run",
        type="primary",
        use_container_width=True,
        key="start_new_run",
    ):
        # Reset session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)