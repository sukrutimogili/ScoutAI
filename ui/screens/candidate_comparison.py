"""
Candidate Comparison — side-by-side, same vocabulary as Candidate Review.

Shared row grid so each section aligns horizontally across candidates.
"""

from __future__ import annotations

import streamlit as st

from ui.components import masthead, status_pill, evidence_list, alert_warn
from ui.mock_data import MOCK_CANDIDATES


def _get_candidates() -> list[dict]:
    """Return candidates from real pipeline_data, falling back to mock data."""
    data = st.session_state.get("pipeline_data", {})
    candidates_raw = data.get("candidates", [])
    if candidates_raw:
        return [c if isinstance(c, dict) else c.model_dump() for c in candidates_raw]
    return list(MOCK_CANDIDATES)


def _get_candidate(candidate_id: str, all_candidates: list[dict], fallback_idx: int = 0) -> dict:
    """Look up a candidate by ID, falling back to the nth candidate in the list."""
    match = next((c for c in all_candidates if c.get("candidate_id") == candidate_id), None)
    if match:
        return match
    return all_candidates[fallback_idx] if fallback_idx < len(all_candidates) else {}


def _evidence_items(evidence_raw: object) -> dict[str, list[str]]:
    """
    Normalise evidence_buckets into {category: [str, ...]} regardless of source shape.

    Real backend: list[dict] with value/source/jd_relevance  → grouped by source
    Mock data:    dict[str, list[str]]                        → used as-is
    """
    if isinstance(evidence_raw, dict):
        return {k: list(v) for k, v in evidence_raw.items()}

    if isinstance(evidence_raw, list):
        grouped: dict[str, list[str]] = {}
        for item in evidence_raw:
            if not isinstance(item, dict):
                continue
            source = item.get("source", "General")
            value = item.get("value", "")
            if value:
                grouped.setdefault(source, []).append(value)
        return grouped

    return {}


def _scorecard_score(value: object) -> float:
    """Extract a numeric score from a ScoreEntry dict or a raw number."""
    if isinstance(value, dict):
        return float(value.get("score", 0))
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def render() -> None:
    """Render the candidate comparison screen."""
    run_name = st.session_state.get("run_name", "Hiring Run")
    masthead(run_name)

    all_candidates = _get_candidates()

    candidate_a_id = st.session_state.get("current_candidate_id", "c001")
    candidate_b_id = st.session_state.get("compare_candidate_id", "c002")

    candidate_a = _get_candidate(candidate_a_id, all_candidates, fallback_idx=0)
    candidate_b = _get_candidate(candidate_b_id, all_candidates, fallback_idx=1)

    if not candidate_a or not candidate_b:
        st.warning("Could not load candidates for comparison.")
        if st.button("Back to list", type="secondary"):
            st.session_state["screen"] = "candidate_list"
            st.rerun()
        return

    st.markdown('<div class="content">', unsafe_allow_html=True)

    st.markdown(
        '<p class="eyebrow">CANDIDATE COMPARISON</p>',
        unsafe_allow_html=True,
    )
    st.markdown("<h1>Compare</h1>", unsafe_allow_html=True)

    # Comparison grid — two columns
    st.markdown('<div class="comparison-grid">', unsafe_allow_html=True)

    for candidate in [candidate_a, candidate_b]:
        uncertainties = candidate.get("remaining_uncertainties", [])
        capabilities = candidate.get("capabilities", {})
        evidence_buckets = _evidence_items(candidate.get("evidence_buckets", []))
        strengths = candidate.get("strengths", [])

        html = f"""
        <div class="comparison-col">
            <div style="margin-bottom:16px;">
                {status_pill(candidate.get("recommendation", "hold"))}
                <h2 style="margin-top:8px;">{candidate.get("candidate_id", "—")}</h2>
            </div>

            <h3 style="font-size:13px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">Strengths</h3>
            <ul style="margin:0 0 16px 0;padding-left:16px;">
        """
        for s in strengths:
            html += f"<li style='font-size:13px;color:var(--muted);'>{s}</li>"
        if not strengths:
            html += "<li style='font-size:13px;color:var(--muted);'>None identified</li>"
        html += "</ul>"

        # Key Differentiators — top 2 scorecard criteria by score
        html += """
            <h3 style="font-size:13px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">Key Differentiators</h3>
            <ul style="margin:0 0 16px 0;padding-left:16px;">
        """
        scorecard = candidate.get("scorecard", {})
        top_skills = sorted(
            scorecard.items(),
            key=lambda x: _scorecard_score(x[1]),
            reverse=True,
        )[:2]
        for skill, score_val in top_skills:
            score_num = _scorecard_score(score_val)
            html += f"<li style='font-size:13px;color:var(--muted);'>{skill}: {score_num:.0f}/100</li>"
        if not top_skills:
            html += "<li style='font-size:13px;color:var(--muted);'>No scorecard data</li>"
        html += "</ul>"

        # Needs Validation
        if uncertainties:
            html += '<h3 style="font-size:13px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">Needs Validation</h3>'
            for u in uncertainties:
                html += f'<div class="alert alert-warn" style="margin-bottom:8px;">{u}</div>'

        # Interview Focus — low-confidence capabilities
        low_conf = [
            name for name, cap in capabilities.items()
            if isinstance(cap, dict) and cap.get("confidence", "high") in ("low", "unknown")
        ]
        focus_areas = (uncertainties + low_conf)[:2]
        if focus_areas:
            html += '<h3 style="font-size:13px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;margin-top:16px;">Interview Focus</h3>'
            for area in focus_areas:
                html += f'<div class="evidence-item">{area}</div>'

        # Supporting Evidence — first 2 categories, 1 item each
        if evidence_buckets:
            html += '<h3 style="font-size:13px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;margin-top:16px;">Supporting Evidence</h3>'
            for category, items in list(evidence_buckets.items())[:2]:
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
