"""
Candidate List — scan, sort, filter, pick one.

Rows with hairline dividers, sort/filter controls, chevron affordance to open.
"""

from __future__ import annotations

import streamlit as st

from ui.components import masthead, status_pill, candidate_row
from ui.mock_data import MOCK_CANDIDATES, MOCK_SHORTLIST


def render() -> None:
    """Render the candidate list screen."""
    run_name = st.session_state.get("run_name", "Hiring Run")
    masthead(run_name)

    st.markdown('<div class="content">', unsafe_allow_html=True)

    st.markdown(
        f'<p class="eyebrow">HIRING RUN — {run_name}</p>',
        unsafe_allow_html=True,
    )
    st.markdown("<h1>Candidates</h1>", unsafe_allow_html=True)

    # Sort and filter controls
    col1, col2 = st.columns(2)
    with col1:
        sort_by = st.selectbox(
            "Sort by",
            options=["Recommendation", "Score", "Name"],
            key="candidate_sort",
        )
    with col2:
        # Filter by recommendation
        all_recs = ["All", "Strong Shortlist", "Interview", "Reject", "Needs Review"]
        filter_rec = st.selectbox(
            "Filter",
            options=all_recs,
            key="candidate_filter",
        )

    # Map filter label to recommendation value
    rec_map = {
        "Strong Shortlist": "strong_interview",
        "Interview": "interview",
        "Reject": "reject",
        "Needs Review": "hold",
    }

    # Build candidate list from mock data
    candidates = list(MOCK_CANDIDATES)

    # Apply filter
    if filter_rec != "All":
        rec_val = rec_map.get(filter_rec)
        candidates = [c for c in candidates if c["recommendation"] == rec_val]

    # Apply sort
    if sort_by == "Score":
        # Sort by average scorecard score descending
        def avg_score(c):
            scores = list(c.get("scorecard", {}).values())
            return sum(scores) / len(scores) if scores else 0

        candidates.sort(key=avg_score, reverse=True)
    elif sort_by == "Name":
        candidates.sort(key=lambda c: c["candidate_id"])
    else:
        # Recommendation — sort by priority
        priority = {"strong_interview": 0, "interview": 1, "hold": 2, "reject": 3}
        candidates.sort(key=lambda c: priority.get(c["recommendation"], 99))

    # Render candidate rows
    st.markdown('<div style="margin-top:16px;">', unsafe_allow_html=True)
    for c in candidates:
        # Build a one-line summary from strengths
        strengths = c.get("strengths", [])
        summary = "; ".join(strengths[:2]) if strengths else "No strengths identified"

        # Get score from shortlist if available
        shortlist_entry = next(
            (s for s in MOCK_SHORTLIST if s["candidate"] == c["candidate_id"]),
            None,
        )
        score = shortlist_entry.get("weighted_score") if shortlist_entry else None

        # Render the row
        st.markdown(
            candidate_row(c["candidate_id"], c["recommendation"], summary, score),
            unsafe_allow_html=True,
        )

        # Invisible button to navigate to candidate review
        if st.button(
            f"Open {c['candidate_id']}",
            key=f"open_{c['candidate_id']}",
            help=f"Review {c['candidate_id']}",
        ):
            st.session_state["current_candidate_id"] = c["candidate_id"]
            st.session_state["screen"] = "candidate_review"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Back to summary
    if st.button("Back to summary", type="secondary", key="back_to_summary"):
        st.session_state["screen"] = "summary"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)