"""
ScoutAI Streamlit UI — monochrome, editorial-grid, keyboard-first hiring review tool.

Navigation between screens is driven by st.session_state["screen"], not a top-nav menu.
Flow order: Landing → Processing → Summary → Candidate List → Candidate Review →
           Candidate Comparison → Continue Reviewing → Hiring Run Complete
"""

from __future__ import annotations

import streamlit as st

from ui.styles import inject_css

# Must be the first Streamlit command
st.set_page_config(
    page_title="ScoutAI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject design tokens and CSS
st.markdown(f"<style>{inject_css()}</style>", unsafe_allow_html=True)

# ── Session state initialization ──────────────────────────────────────────────

if "screen" not in st.session_state:
    st.session_state["screen"] = "landing"

# ── Screen routing ────────────────────────────────────────────────────────────

screen = st.session_state["screen"]

if screen == "landing":
    from ui.screens.landing import render as render_landing
    render_landing()

elif screen == "processing":
    from ui.screens.processing import render as render_processing
    render_processing()

elif screen == "summary":
    from ui.screens.hiring_run_summary import render as render_summary
    render_summary()

elif screen == "candidate_list":
    from ui.screens.candidate_list import render as render_candidate_list
    render_candidate_list()

elif screen == "candidate_review":
    from ui.screens.candidate_review import render as render_candidate_review
    render_candidate_review()

elif screen == "candidate_comparison":
    from ui.screens.candidate_comparison import render as render_candidate_comparison
    render_candidate_comparison()

elif screen == "continue_reviewing":
    from ui.screens.continue_reviewing import render as render_continue_reviewing
    render_continue_reviewing()

elif screen == "complete":
    from ui.screens.hiring_run_complete import render as render_hiring_run_complete
    render_hiring_run_complete()

else:
    # Fallback to landing
    st.session_state["screen"] = "landing"
    st.rerun()
