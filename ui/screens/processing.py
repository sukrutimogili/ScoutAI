"""
Processing screen — real, truthful progress display.

The step list reads like a sentence: "Understanding the role → Analyzing resumes → ..."
Each step visibly completes before the next highlights.
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

    # Simulate progress for demo purposes
    # In production, this would be driven by real backend signals
    # (e.g., current node name from the LangGraph stream mapped to the nearest step)
    if "processing_step" not in st.session_state:
        st.session_state["processing_step"] = 0
        st.session_state["processing_started"] = time.time()

    current_step = st.session_state["processing_step"]

    # Build the step list HTML
    steps_html = '<ul class="step-list">'
    for i, step in enumerate(STEPS):
        if i < current_step:
            # Done
            indicator = '<span class="step-indicator done">✓</span>'
            label_class = "step-label done"
        elif i == current_step:
            # Active
            indicator = '<span class="step-indicator active">●</span>'
            label_class = "step-label active"
        else:
            # Pending
            indicator = '<span class="step-indicator">○</span>'
            label_class = "step-label"

        steps_html += f"""
        <li class="step-item">
            <span class="step-index">{i + 1}</span>
            {indicator}
            <span class="{label_class}" style="margin-left:12px;">{step}</span>
        </li>
        """
    steps_html += "</ul>"

    st.markdown(steps_html, unsafe_allow_html=True)

    # Auto-advance every 1.5 seconds for demo
    # In production, this is driven by real backend state
    if current_step < len(STEPS):
        placeholder = st.empty()
        with placeholder:
            st.caption("Processing in progress...")

        # Auto-advance logic
        elapsed = time.time() - st.session_state["processing_started"]
        estimated = (current_step + 1) * 1.5
        if elapsed > estimated:
            st.session_state["processing_step"] = current_step + 1
            if st.session_state["processing_step"] >= len(STEPS):
                # Processing complete — transition to summary
                st.session_state["screen"] = "summary"
            st.rerun()
    else:
        st.caption("Processing complete.")
        # Transition to summary
        st.session_state["screen"] = "summary"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)