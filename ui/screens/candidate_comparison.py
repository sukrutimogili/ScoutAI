"""
Candidate Comparison — side-by-side, same vocabulary as Candidate Review.

Shared row grid so each section aligns horizontally across candidates.
"""

from __future__ import annotations

import streamlit as st

from ui.components import masthead, status_pill, evidence_list, alert_warn
from ui.mock_data import MOCK_CANDIDATES


def render() -> None:
    """Render the candidate comparison screen."""
    run_name = st.session_state.get("run_name", "Hiring Run")
    masthead(run_name)

    candidate_a_id = st.session_state.get("current_candidate_id", "c001")
    candidate_b_id = st.session_state.get("compare_candidate_id", "c002")

    candidate_a = next(
        (c for c in MOCK_CANDIDATES if c["candidate_id"] == candidate_a_id),
        MOCK_CANDIDATES[0],
    )
    candidate_b = next(
        (c for c in MOCK_CANDIDATES if c["candidate_id"] == candidate_b_id),
        MOCK_CANDIDATES[1],
    )

    st.markdown('<div class="content">', unsafe_allow_html=True)

    st.markdown(
        f'<p class="eyebrow">CANDIDATE COMPARISON</p>',
        unsafe_allow_html=True,
    )
    st.markdown("<h1>Compare</h1>", unsafe_allow_html=True)

    # Comparison grid — two columns
    st.markdown('<div class="comparison-grid">', unsafe_allow_html=True)

    for candidate in [candidate_a, candidate_b]:
        uncertainties = candidate.get("remaining_uncertainties", [])
        capabilities = candidate.get("capabilities", {})
        evidence = candidate.get("evidence_buckets", {})
        strengths = candidate.get("strengths", [])

        html = f"""
        <div class="comparison-col">
            <div style="margin-bottom:16px;">
                {status_pill(candidate["recommendation"])}
                <h2 style="margin-top:8px;">{candidate["candidate_id"]}</h2>
            </div>

            <h3 style="font-size:13px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">Strengths</h3>
            <ul style="margin:0 0 16px 0;padding-left:16px;">
        """
        for s in strengths:
            html += f"<li style='font-size:13px;color:var(--muted);'>{s}</li>"
        html += "</ul>"

        # Key Differentiators
        html += """
            <h3 style="font-size:13px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">Key Differentiators</h3>
            <ul style="margin:0 0 16px 0;padding-left:16px;">
        """
        # Use scorecard differences as differentiators
        scorecard = candidate.get("scorecard", {})
        for skill, score in sorted(scorecard.items(), key=lambda x: x[1], reverse=True)[:2]:
            html += f"<li style='font-size:13px;color:var(--muted);'>{skill}: {score}/100</li>"
        html += "</ul>"

        # Needs Validation
        if uncertainties:
            html += f"""
            <h3 style="font-size:13px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">Needs Validation</h3>
            """
            for u in uncertainties:
                html += f'<div class="alert alert-warn" style="margin-bottom:8px;">{u}</div>'

        # Interview Focus
        low_conf = [
            name for name, cap in capabilities.items()
            if cap.get("confidence", "high") in ("low", "unknown")
        ]
        focus_areas = uncertainties + low_conf
        if focus_areas:
            html += f"""
            <h3 style="font-size:13px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;margin-top:16px;">Interview Focus</h3>
            """
            for area in focus_areas[:2]:
                html += f'<div class="evidence-item">{area}</div>'

        # Supporting Evidence (first 2 items only for comparison)
        if evidence:
            html += f"""
            <h3 style="font-size:13px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;margin-top:16px;">Supporting Evidence</h3>
            """
            for category, items in list(evidence.items())[:2]:
                html += f'<p style="font-size:12px;font-weight:700;color:var(--muted-2);margin:8px 0 4px 0;">{category}</p>'
                for item in items[:1]:
                    html += f'<div class="evidence-item">{item}</div>'

        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Back to review
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Back to review", type="secondary", key="back_to_review"):
            st.session_state["screen"] = "candidate_review"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)