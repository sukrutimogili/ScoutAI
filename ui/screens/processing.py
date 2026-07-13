"""
Processing screen — polls the pipeline background thread and transitions
to the summary screen when the run completes or pauses at human_review.

Threading contract (see landing.py for the producer side)
---------------------------------------------------------
  st.session_state["_pipeline_done_event"]   threading.Event
  st.session_state["_pipeline_result_bag"]   plain dict written by the thread

The main Streamlit thread (this screen) only ever reads the event/bag.
It never writes to session_state from a background thread, which is how
we avoid Streamlit's silent write-drop bug.
"""

from __future__ import annotations

import time
import threading

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

    done_event: threading.Event | None = st.session_state.get("_pipeline_done_event")
    result_bag: dict = st.session_state.get("_pipeline_result_bag", {})
    thread: threading.Thread | None = st.session_state.get("_pipeline_thread")

    # ── Check if pipeline finished ─────────────────────────────────────────────
    pipeline_finished = done_event is not None and done_event.is_set()

    if pipeline_finished:
        if result_bag.get("ok"):
            # Success — move results into session_state on the main thread
            st.session_state["pipeline_data"] = result_bag["data"]
            # Clean up processing-only state
            for key in ("processing_step", "processing_started",
                        "_pipeline_done_event", "_pipeline_result_bag", "_pipeline_thread"):
                st.session_state.pop(key, None)
            st.session_state["screen"] = "summary"
            st.rerun()
            return
        else:
            # Error — show it
            st.error(result_bag.get("error", "An unknown error occurred."))
            detail = result_bag.get("detail", "")
            if detail:
                with st.expander("Technical details"):
                    st.code(detail, language="text")
            if st.button("Back to start", type="primary", key="proc_back_error"):
                for key in ("_pipeline_done_event", "_pipeline_result_bag",
                            "_pipeline_thread", "processing_step", "processing_started"):
                    st.session_state.pop(key, None)
                st.session_state["screen"] = "landing"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
            return

    # ── Thread-death guard ─────────────────────────────────────────────────────
    if thread is not None and not thread.is_alive() and not pipeline_finished:
        st.error(
            "The pipeline stopped unexpectedly before producing results. "
            "Please try again."
        )
        if st.button("Back to start", type="primary", key="proc_back_dead"):
            st.session_state["screen"] = "landing"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if done_event is None:
        # Navigated to this screen without starting a run
        st.warning("No pipeline run in progress.")
        if st.button("Start a run", type="primary"):
            st.session_state["screen"] = "landing"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ── Animated progress indicator ───────────────────────────────────────────
    if "processing_step" not in st.session_state:
        st.session_state["processing_step"] = 0
        st.session_state["processing_started"] = time.time()

    current_step = st.session_state["processing_step"]
    elapsed = time.time() - st.session_state["processing_started"]

    # Advance one step every 2 s while the thread is still running
    if elapsed > (current_step + 1) * 2.0 and current_step < len(STEPS):
        st.session_state["processing_step"] = current_step + 1
        st.rerun()

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

    label = "Processing in progress..." if current_step < len(STEPS) else "Finalizing results..."
    st.caption(label)

    # Poll every second — done_event.is_set() is the authoritative signal
    time.sleep(1)
    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
