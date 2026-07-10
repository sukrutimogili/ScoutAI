"""
Landing screen — upload JD, résumés, start the run.

Three inputs, vertically stacked, generously spaced. One primary action.
"""

from __future__ import annotations

import streamlit as st

from ui.components import masthead


def render() -> None:
    """Render the landing screen."""
    masthead()

    st.markdown('<div class="content">', unsafe_allow_html=True)

    st.markdown('<p class="eyebrow">NEW HIRING RUN</p>', unsafe_allow_html=True)
    st.markdown("<h1>Start a hiring run</h1>", unsafe_allow_html=True)

    st.markdown('<div style="margin-top:32px;">', unsafe_allow_html=True)

    # Job description
    jd = st.text_area(
        "Job description",
        placeholder="Paste the job description here, or upload a file below...",
        height=200,
        key="landing_jd",
    )

    # Résumé upload
    resumes = st.file_uploader(
        "Résumés",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt"],
        key="landing_resumes",
        help="PDF, DOCX, or TXT",
    )

    # Run name (optional)
    run_name = st.text_input(
        "Run name",
        placeholder="Optional — defaults to the date and role title",
        key="landing_run_name",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # Primary action
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        disabled = not (jd.strip() and resumes)
        if st.button(
            "Start evaluation",
            type="primary",
            use_container_width=True,
            disabled=disabled,
            key="landing_start",
        ):
            # Store the run data in session state
            st.session_state["run_name"] = run_name or "Untitled Run"
            st.session_state["jd"] = jd
            st.session_state["resumes"] = resumes
            st.session_state["screen"] = "processing"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)