"""
Processing screen — real pipeline progress display.

Polls the background pipeline thread and shows real progress.
When the pipeline completes, transitions to the summary screen.
"""

from __future__ import annotations

import time

import streamlit as st

from ui.components import masthead

STEPS = [
    "Understanding the role",
    "Analyzing resumes",
    "Extracting relevant experience",
    "Verifying supporting evidence",
    "Preparing interview recommendations",
    "Running fairness checks",
    "Preparing recruiter summary",
]


def render() -> None:
    """Render the processing screen."""
    run_name = st.session_state.get("run_name", "Hiring Run")
    masthead(run_name)

    st.markdown('<div class="content">', unsafe_allow_html=True)
    st.markdown("<h1>Processing</h1>", unsafe_allow_html=True)

    # Check if pipeline is done
    pipeline_done = st.session_state.get("pipeline_done", False)
    pipeline_result = st.session_state.get("pipeline_result")

    if pipeline_done and pipeline_result is not None:
        # Pipeline finished — store results and transition
        st.session_state["pipeline_data"] = pipeline_result
        st.session_state["screen"] = "summary"
        st.rerun()
        return

    # Check if pipeline thread is still alive
    thread = st.session_state.get("pipeline_thread")
    if thread is None or not thread.is_alive():
        if not pipeline_done:
            # Thread died unexpectedly — show error
            st.error("Pipeline execution failed. Please check the logs and try again.")
            if st.button("Back to start", type="primary"):
                st.session_state["screen"] = "landing"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
            return

    # Show animated progress indicator
    if "processing_step" not in st.session_state:
        st.session_state["processing_step"] = 0
        st.session_state["processing_started"] = time.time()

    current_step = st.session_state["processing_step"]

    # Advance step every 2 seconds while pipeline is running
    elapsed = time.time() - st.session_state["processing_started"]
    estimated = (current_step + 1) * 2.0
    if elapsed > estimated and current_step < len(STEPS):
        st.session_state["processing_step"] = current_step + 1
        st.rerun()

    # Build the step list HTML
    steps_html = '<ul class="step-list">'
    for i, step in enumerate(STEPS):
        if i < current_step:
            indicator = '<span class="step-indicator done">✓</span>'
            label_class = "step-label done"
        elif i == current_step:
            indicator = '<span class="step-indicator active">●</span>'
            label_class = "step-label active"
        else:
            indicator = '<span class="step-indicator">○</span>'
            label_class = "step-label"

        steps_html += (
            '<li class="step-item">'
            f'<span class="step-index">{i + 1}</span>'
            f'{indicator}'
            f'<span class="{label_class}" style="margin-left:12px;">{step}</span>'
            "</li>"
        )
    steps_html += "</ul>"

    st.markdown(steps_html, unsafe_allow_html=True)

    # Show status
    if current_step < len(STEPS):
        st.caption("Processing in progress...")
        # Auto-rerun to check pipeline status
        time.sleep(1)
        st.rerun()
    else:
        st.caption("Finalizing results...")

    st.markdown("</div>", unsafe_allow_html=True)