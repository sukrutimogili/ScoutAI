"""
Continue Reviewing — transition state after an action on Candidate Review.

Shows a brief confirmation, then "Next candidate" as the primary action,
with the review progress bar.
"""

from __future__ import annotations

import streamlit as st

from ui.components import masthead, progress_bar, alert_warn
from ui.mock_data import MOCK_CANDIDATES


def _get_candidates() -> list[dict]:
    """Return candidates from real pipeline_data, falling back to mock data."""
    data = st.session_state.get("pipeline_data", {})
    candidates_raw = data.get("candidates", [])
    if candidates_raw:
        return [c if isinstance(c, dict) else c.model_dump() for c in candidates_raw]
    return list(MOCK_CANDIDATES)


def render() -> None:
    """Render the continue reviewing screen."""
    run_name = st.session_state.get("run_name", "Hiring Run")
    masthead(run_name)

    last_action = st.session_state.get("last_action", "processed")
    action_verb = {
        "approved": "Approved",
        "rejected": "Rejected",
        "skipped": "Skipped",
    }.get(last_action, "Processed")

    st.markdown('<div class="content">', unsafe_allow_html=True)

    # Confirmation
    st.markdown(
        f'<p class="display-number" style="font-size:28px;">{action_verb}</p>',
        unsafe_allow_html=True,
    )
    candidate_id = st.session_state.get("current_candidate_id", "?")
    st.markdown(
        f'<p class="caption" style="margin-top:4px;margin-bottom:32px;">{candidate_id} — {action_verb.lower()}</p>',
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Review progress — use real candidate list for an accurate total
    candidates = _get_candidates()
    total = len(candidates)
    reviewed = st.session_state.get("reviewed_count", 0)
    reviewed += 1
    st.session_state["reviewed_count"] = reviewed

    st.markdown(
        f'<p class="caption" style="margin-bottom:8px;">{reviewed} of {total} reviewed</p>',
        unsafe_allow_html=True,
    )
    st.markdown(progress_bar(reviewed, total), unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Find next un-reviewed candidate (sequential by current list order)
    current_idx = next(
        (i for i, c in enumerate(candidates) if c.get("candidate_id") == candidate_id),
        0,
    )
    next_idx = current_idx + 1

    if next_idx < total:
        # Next candidate
        if st.button(
            "Next candidate",
            type="primary",
            use_container_width=True,
            key="next_candidate",
        ):
            st.session_state["current_candidate_id"] = candidates[next_idx]["candidate_id"]
            st.session_state["screen"] = "candidate_review"
            st.rerun()
    else:
        # All reviewed — go to complete
        st.markdown(
            '<p class="caption" style="margin-bottom:16px;">All candidates reviewed.</p>',
            unsafe_allow_html=True,
        )
        if st.button(
            "View final summary",
            type="primary",
            use_container_width=True,
            key="view_final",
        ):
            st.session_state["screen"] = "complete"
            st.rerun()

    # Secondary action: back to list
    if st.button("Back to list", type="secondary", use_container_width=True, key="continue_back_to_list"):
        st.session_state["screen"] = "candidate_list"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
