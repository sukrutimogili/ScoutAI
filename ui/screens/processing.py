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

from ui.components import masthead, alert_warn

STEPS = [
    "Understanding the role",
    "Analyzing resumes",
    "Extracting relevant experience",
    "Verifying supporting evidence",
    "Preparing interview recommendations",
    "Running fairness checks",
    "Preparing recruiter summary",
]


def _has_no_evaluated_candidates(data: dict) -> bool:
    """True if the pipeline produced candidates but none of them were scored.

    This is the signature of a run that paused at human_review before
    candidate_agent ever ran — every candidate was fail-closed at screening
    (e.g. because the LLM call failed) and none has a recommendation yet.
    """
    candidates = data.get("candidates", []) if data else []
    if not candidates:
        return False
    for c in candidates:
        rec = c.get("recommendation") if isinstance(c, dict) else getattr(c, "recommendation", None)
        if rec is not None:
            return False
    return True


def render() -> None:
    """Render the processing screen."""
    run_name = st.session_state.get("run_name", "Hiring Run")
    masthead(run_name)

    st.markdown('<div class="content">', unsafe_allow_html=True)
    st.markdown("<h1>Processing</h1>", unsafe_allow_html=True)

    _poll_pipeline()

    st.markdown("</div>", unsafe_allow_html=True)


@st.fragment(run_every=1)
def _poll_pipeline() -> None:
    """
    Poll the background pipeline thread once a second.

    This body used to live directly in render() and call `time.sleep(1)` then
    a bare `st.rerun()` every second. A bare st.rerun() reruns the *entire*
    Streamlit script from the top of app.py — re-firing st.set_page_config,
    re-injecting the whole <style> block, and tearing down/rebuilding the DOM
    every second. That's what caused the browser tab title to flicker
    ("ScoutAI" ↔ "Streamlit") and made screens look like they were bleeding
    into each other — the routing in app.py was fine, you were just seeing
    the whole app reconstruct itself once a second.

    `st.fragment(run_every=1)` reruns *only this function* every second — the
    rest of app.py (page config, CSS, masthead) is untouched. We only ever
    force a full-app rerun (scope="app") when we're actually navigating to a
    different screen.
    """
    done_event: threading.Event | None = st.session_state.get("_pipeline_done_event")
    result_bag: dict = st.session_state.get("_pipeline_result_bag", {})
    thread: threading.Thread | None = st.session_state.get("_pipeline_thread")

    # ── Check if pipeline finished ─────────────────────────────────────────────
    pipeline_finished = done_event is not None and done_event.is_set()

    if pipeline_finished:
        if result_bag.get("ok"):
            data = result_bag.get("data", {}) or {}
            paused = result_bag.get("paused", False)
            no_evaluations = _has_no_evaluated_candidates(data)

            if paused and no_evaluations:
                # The run paused at human_review before any candidate was ever
                # scored — most commonly because every résumé fail-closed at
                # screen_resume (e.g. LLM provider unavailable / missing key).
                # Don't silently land on an empty "0 of N" summary — say so.
                st.session_state["pipeline_data"] = data
                for key in ("processing_step", "processing_started",
                            "_pipeline_done_event", "_pipeline_result_bag", "_pipeline_thread"):
                    st.session_state.pop(key, None)

                st.markdown(
                    alert_warn(
                        "Evaluation could not complete — all résumés were flagged "
                        "for review before scoring. This usually means the AI model "
                        "call failed (check your API keys / provider configuration) "
                        "or every résumé independently triggered a security flag."
                    ),
                    unsafe_allow_html=True,
                )
                st.caption(
                    "The run paused at human review with 0 of "
                    f"{len(data.get('candidates', []))} candidates evaluated."
                )
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Review flagged résumés", type="primary",
                                 use_container_width=True, key="proc_review_flagged"):
                        st.session_state["screen"] = "candidate_list"
                        st.rerun(scope="app")
                with col2:
                    if st.button("Back to start", type="secondary",
                                 use_container_width=True, key="proc_back_paused"):
                        st.session_state["screen"] = "landing"
                        st.rerun(scope="app")
                return

            # Success — move results into session_state on the main thread
            st.session_state["pipeline_data"] = data
            # Clean up processing-only state
            for key in ("processing_step", "processing_started",
                        "_pipeline_done_event", "_pipeline_result_bag", "_pipeline_thread"):
                st.session_state.pop(key, None)
            st.session_state["screen"] = "summary"
            st.rerun(scope="app")
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
                st.rerun(scope="app")
            return

    # ── Thread-death guard ─────────────────────────────────────────────────────
    if thread is not None and not thread.is_alive() and not pipeline_finished:
        st.error(
            "The pipeline stopped unexpectedly before producing results. "
            "Please try again."
        )
        if st.button("Back to start", type="primary", key="proc_back_dead"):
            st.session_state["screen"] = "landing"
            st.rerun(scope="app")
        return

    if done_event is None:
        # Navigated to this screen without starting a run
        st.warning("No pipeline run in progress.")
        if st.button("Start a run", type="primary"):
            st.session_state["screen"] = "landing"
            st.rerun(scope="app")
        return

    # ── Animated progress indicator ───────────────────────────────────────────
    if "processing_step" not in st.session_state:
        st.session_state["processing_step"] = 0
        st.session_state["processing_started"] = time.time()

    elapsed = time.time() - st.session_state["processing_started"]

    # Advance one step every 2 s while the thread is still running. The
    # fragment already reruns this function every second on its own (via
    # run_every=1 below), so we just recompute how many steps should be
    # "done" from elapsed time rather than calling st.rerun() ourselves.
    current_step = min(int(elapsed // 2.0), len(STEPS))
    st.session_state["processing_step"] = current_step

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

    # No manual sleep/rerun here — the @st.fragment(run_every=1) decorator on
    # this function is the poll. done_event.is_set() is the authoritative
    # signal for whether the pipeline has finished, checked at the top of
    # this function on every 1s tick.