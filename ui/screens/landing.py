"""
Landing screen — upload JD, résumés, start the run.

Three inputs, vertically stacked, generously spaced. One primary action.

Threading note
--------------
Streamlit session_state cannot be written from a background thread — writes are
silently dropped, leaving the processing screen stuck forever.  We communicate
between the pipeline thread and the Streamlit main thread using plain Python
objects stored inside session_state:

  st.session_state["_pipeline_event"]   threading.Event — set when done/error
  st.session_state["_pipeline_result"]  dict — {"ok": True,  "data": <state>}
                                              | {"ok": False, "error": str,
                                                             "detail": str}

The main thread only reads from session_state; the background thread only writes
to the shared plain dict and then sets the Event.  The processing screen polls
the Event and copies results back into session_state on the main thread.
"""

from __future__ import annotations

import threading
import traceback

import streamlit as st

from ui.components import masthead


def _friendly_error_message(exc: Exception) -> str:
    """Translate common exceptions into recruiter-readable text."""
    try:
        from scoutai.capabilities.model_router import ProviderError
        if isinstance(exc, ProviderError):
            return (
                "The AI model providers are all currently unavailable "
                "(rate-limited or unreachable). Please try again in a few minutes."
            )
    except ImportError:
        pass
    if isinstance(exc, EnvironmentError):
        return (
            "ScoutAI isn't fully configured — an API key is missing. "
            "Contact your ScoutAI administrator."
        )
    return (
        "Something went wrong during analysis. Your JD and résumés were not lost — "
        f"please try again. Error: {type(exc).__name__}: {exc}"
    )


def _run_pipeline(
    thread_id: str,
    jd_text: str,
    resume_texts: list[str],
    result_bag: dict,
    done_event: threading.Event,
) -> None:
    """
    Run the full pipeline and signal completion via done_event.

    All results (success or error) are written to result_bag — a plain dict
    shared with the main Streamlit thread.  done_event.set() is the last thing
    called so the main thread never reads a partial result_bag.
    """
    try:
        from scoutai.runtime import session as _session
        from scoutai.schemas import CandidateState
        from scoutai.graph.app import run_graph

        _session._ensure_initialized()

        candidates = [
            CandidateState(
                candidate_id=f"c{i + 1:03d}",
                resume_text=text,
                sanitized_resume=text,
                injection_flag=False,
                leakage_flag=False,
                finalized=False,
            )
            for i, text in enumerate(resume_texts)
        ]

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

        result = run_graph(
            _session._graph, initial_state, _session._config, thread_id=thread_id
        )

        # Check whether the run paused at a human_review interrupt
        state_snapshot = _session._graph.get_state(
            {"configurable": {"thread_id": thread_id}}
        )
        pending = list(state_snapshot.next) if state_snapshot and state_snapshot.next else []
        paused = bool(pending)

        result_bag["ok"] = True
        result_bag["data"] = result
        result_bag["paused"] = paused
        result_bag["pending_nodes"] = pending

    except Exception as exc:  # noqa: BLE001
        result_bag["ok"] = False
        result_bag["error"] = _friendly_error_message(exc)
        result_bag["detail"] = traceback.format_exc()

    finally:
        done_event.set()


def render() -> None:
    """Render the landing screen with a premium two-column tech grid."""
    # masthead()

    # 1. Open the primary column split engine (Matches our CSS rules perfectly)
    col1, col2 = st.columns(2)

    with col1:
        # Editorial visual block elements
        st.markdown('<p class="eyebrow">NEW HIRING RUN</p>', unsafe_allow_html=True)
        st.markdown("<h1>Start a<br>hiring run</h1>", unsafe_allow_html=True)
        
        # Grounding technical structural SVG illustration to balance empty white space
        st.markdown(
            """
            <div style="margin-top: 64px; max-width: 340px;">
                <svg width="100%" height="160" viewBox="0 0 320 160" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <line x1="0" y1="20" x2="320" y2="20" stroke="#0A0A0A" stroke-width="0.75" stroke-dasharray="4 4" opacity="0.2"/>
                    <line x1="0" y1="140" x2="320" y2="140" stroke="#0A0A0A" stroke-width="0.75" stroke-dasharray="4 4" opacity="0.2"/>
                    <rect x="20" y="45" width="70" height="70" stroke="#0A0A0A" stroke-width="1.25" fill="none"/>
                    <circle cx="160" cy="80" r="35" stroke="#0A0A0A" stroke-width="1.25" fill="none"/>
                    <path d="M240 80 L290 80 M265 55 L265 105" stroke="#0A0A0A" stroke-width="1.25"/>
                    <text x="20" y="132" font-family="Space Mono, monospace" font-size="9" fill="#6B6B6B" letter-spacing="0.05em">SYS.EVAL // KEY_01</text>
                </svg>
            </div>
            """, 
            unsafe_allow_html=True
        )

    with col2:
        # 2. Interactive user inputs aligned to the clean right workspace column
        jd = st.text_area(
            "Job description",
            placeholder="Paste the job description here...",
            height=200,
            key="landing_jd",
        )

        resumes = st.file_uploader(
            "Résumés",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt"],
            key="landing_resumes",
            help="PDF, DOCX, or TXT",
        )

        run_name = st.text_input(
            "Run name",
            placeholder="Optional — defaults to the date and role title",
            key="landing_run_name",
        )

        # Action execution triggers perfectly wide beneath the forms
        disabled = not (jd.strip() and resumes)
        if st.button(
            "Start evaluation",
            type="primary",
            use_container_width=True,
            disabled=disabled,
            key="landing_start",
        ):
            resume_texts = [
                f.read().decode("utf-8", errors="replace") for f in resumes
            ]

            thread_id = f"ui_run_{hash(jd) & 0xFFFFFFFF:08x}"

            # Shared communication objects
            result_bag: dict = {}
            done_event = threading.Event()

            # Store states for the processing module
            st.session_state["run_name"] = run_name or "Untitled Run"
            st.session_state["thread_id"] = thread_id
            st.session_state["_pipeline_result_bag"] = result_bag
            st.session_state["_pipeline_done_event"] = done_event
            
            # Clear historical caches cleanly
            for key in ("pipeline_data", "pipeline_error", "pipeline_error_detail",
                        "processing_step", "processing_started", "reviewed_count"):
                st.session_state.pop(key, None)

            thread = threading.Thread(
                target=_run_pipeline,
                args=(thread_id, jd, resume_texts, result_bag, done_event),
                daemon=True,
            )
            st.session_state["_pipeline_thread"] = thread
            thread.start()

            st.session_state["screen"] = "processing"
            st.rerun()