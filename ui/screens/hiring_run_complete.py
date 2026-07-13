"""
Hiring Run Complete — final summary with export options.

Mirrors the Hiring Run Summary layout, relabeled as final.
"""

from __future__ import annotations

import streamlit as st

from ui.components import masthead, stat_block, progress_bar, alert_warn
from ui.mock_data import MOCK_CANDIDATES, MOCK_BIAS_REPORTS


def _get_candidates() -> list[dict]:
    """Return candidates from real pipeline_data, falling back to mock data."""
    data = st.session_state.get("pipeline_data", {})
    candidates_raw = data.get("candidates", [])
    if candidates_raw:
        return [c if isinstance(c, dict) else c.model_dump() for c in candidates_raw]
    return list(MOCK_CANDIDATES)


def _get_bias_reports() -> list[dict]:
    """Return bias reports from real pipeline_data, falling back to mock data."""
    data = st.session_state.get("pipeline_data", {})
    reports_raw = data.get("bias_reports", [])
    if reports_raw:
        return [r if isinstance(r, dict) else r.model_dump() for r in reports_raw]
    return list(MOCK_BIAS_REPORTS)


def _pipeline_state() -> dict:
    """Return the pipeline_data dict, falling back to a minimal mock state."""
    data = st.session_state.get("pipeline_data")
    if data:
        return data
    # Build a minimal state from mock data so exports always work
    from ui.mock_data import MOCK_CANDIDATES, MOCK_BIAS_REPORTS, MOCK_SHORTLIST
    return {
        "candidates": MOCK_CANDIDATES,
        "bias_reports": MOCK_BIAS_REPORTS,
        "shortlist": MOCK_SHORTLIST,
        "run_id": st.session_state.get("run_name", "demo-run"),
        "step_count": 0,
    }


def render() -> None:
    """Render the hiring run complete screen."""
    run_name = st.session_state.get("run_name", "Hiring Run")
    masthead(run_name)

    st.markdown('<div class="content">', unsafe_allow_html=True)

    # Load from pipeline_data with mock fallback
    candidates = _get_candidates()
    bias_reports = _get_bias_reports()

    # Count recommendations
    total = len(candidates)
    strong = sum(1 for c in candidates if c.get("recommendation") == "strong_interview")
    interview = sum(1 for c in candidates if c.get("recommendation") == "interview")
    reject = sum(1 for c in candidates if c.get("recommendation") == "reject")
    needs_review = sum(1 for c in candidates if c.get("recommendation") == "hold")

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

    # Export section
    st.markdown("<h2>Export</h2>", unsafe_allow_html=True)
    st.markdown(
        '<p class="caption" style="margin-bottom:16px;">Download the full report in your preferred format.</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("PDF", type="secondary", use_container_width=True, key="export_pdf"):
            try:
                from scoutai.graph.export import export_to_pdf
                import tempfile, os
                state = _pipeline_state()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                    export_to_pdf(state, tmp.name)
                    tmp_path = tmp.name
                with open(tmp_path, "rb") as f:
                    pdf_bytes = f.read()
                os.unlink(tmp_path)
                st.download_button(
                    label="Download PDF report",
                    data=pdf_bytes,
                    file_name=f"{run_name.replace(' ', '_')}_report.txt",
                    mime="text/plain",
                    key="download_pdf",
                )
            except Exception as exc:
                st.error(f"Export failed: {exc}")

    with col2:
        if st.button("CSV", type="secondary", use_container_width=True, key="export_csv"):
            try:
                from scoutai.graph.export import export_candidates_to_csv, export_shortlist_to_csv
                state = _pipeline_state()
                csv_data = export_candidates_to_csv(state)
                shortlist_csv = export_shortlist_to_csv(state)
                combined = "# CANDIDATES\n" + csv_data + "\n# SHORTLIST\n" + shortlist_csv
                st.download_button(
                    label="Download CSV",
                    data=combined.encode("utf-8"),
                    file_name=f"{run_name.replace(' ', '_')}_candidates.csv",
                    mime="text/csv",
                    key="download_csv",
                )
            except Exception as exc:
                st.error(f"Export failed: {exc}")

    with col3:
        if st.button("JSON", type="secondary", use_container_width=True, key="export_json"):
            try:
                from scoutai.graph.export import export_to_json
                state = _pipeline_state()
                json_str = export_to_json(state)
                st.download_button(
                    label="Download JSON",
                    data=json_str.encode("utf-8"),
                    file_name=f"{run_name.replace(' ', '_')}_state.json",
                    mime="application/json",
                    key="download_json",
                )
            except Exception as exc:
                st.error(f"Export failed: {exc}")

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
