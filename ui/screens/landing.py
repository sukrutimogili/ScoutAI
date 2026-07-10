"""
Landing screen — upload JD, résumés, start the run.

Three inputs, vertically stacked, generously spaced. One primary action.
"""

from __future__ import annotations

import threading
from pathlib import Path

import streamlit as st

from scoutai.config import load_config
from scoutai.runtime.session import start_run
from ui.components import masthead


def _run_pipeline(thread_id: str, jd_text: str, resume_texts: list[str]) -> None:
    """Run the pipeline in a background thread and store results."""
    from scoutai.runtime import session as _session

    _session._ensure_initialized()

    # Build initial state
    from scoutai.schemas import CandidateState
    candidates = []
    for i, text in enumerate(resume_texts):
        candidates.append(
            CandidateState(
                candidate_id=f"c{i+1:03d}",
                resume_text=text,
                sanitized_resume=text,
                injection_flag=False,
                leakage_flag=False,
                finalized=False,
            )
        )

    initial_state = {
        "jd": jd_text,
        "candidates": candidates,
        "current_idx": 0,
        "shortlist": [],
        "trajectory": [],
        "step_count": 0,
        "run_id": thread_id,
        "role_profile": None,
        "rubric": None,
    }

    from scoutai.graph.app import run_graph
    result = run_graph(_session._graph, initial_state, _session._config, thread_id=thread_id)

    # Store results in session state
    st.session_state["pipeline_result"] = result
    st.session_state["pipeline_done"] = True


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
            st.session_state["pipeline_done"] = False
            st.session_state["pipeline_result"] = None
            st.session_state["screen"] = "processing"

            # Read resume texts
            resume_texts = []
            for f in resumes:
                text = f.read().decode("utf-8", errors="replace")
                resume_texts.append(text)

            # Start pipeline in background thread
            thread_id = f"ui_run_{hash(jd) & 0xFFFFFFFF:08x}"
            st.session_state["thread_id"] = thread_id
            thread = threading.Thread(
                target=_run_pipeline,
                args=(thread_id, jd, resume_texts),
                daemon=True,
            )
            thread.start()
            st.session_state["pipeline_thread"] = thread

            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)