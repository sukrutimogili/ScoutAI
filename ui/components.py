"""
Reusable UI components for ScoutAI Streamlit UI.

Components are built as simple HTML strings rendered via st.markdown.
This keeps them fully controllable with the design token system.
"""

from __future__ import annotations

import streamlit as st

from ui.mock_data import get_recommendation_label, get_recommendation_tint


def masthead(run_name: str | None = None) -> None:
    """Render the persistent masthead bar."""
    right_html = ""
    if run_name:
        right_html = f"""
        <span class="masthead-right">
            {run_name}
            <a onclick="window.location.reload()">Exit run</a>
        </span>
        """
    st.markdown(
        f"""
        <div class="masthead">
            <span class="masthead-wordmark">SCOUTAI</span>
            {right_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_pill(recommendation: str | None) -> str:
    """Return HTML for a recommendation status pill.

    The pill uses tint background only — text is always --ink.
    This keeps the page monochrome at a glance and legible for colorblind users.
    """
    if not recommendation:
        return '<span class="status-pill hold">Pending</span>'
    label = get_recommendation_label(recommendation)
    return f'<span class="status-pill {recommendation}">{label}</span>'


def stat_block(number: str | int, label: str) -> str:
    """Return HTML for a stat block (number + label)."""
    return f"""
    <div class="stat-block">
        <span class="stat-block-number">{number}</span>
        <span class="stat-block-label">{label}</span>
    </div>
    """


def progress_bar(current: int, total: int, label: str = "") -> str:
    """Return HTML for a progress bar.

    current: number of completed items.
    total: total number of items.
    label: optional text label shown below the bar.
    """
    pct = (current / total * 100) if total > 0 else 0
    html = f"""
    <div class="progress-bar">
        <div class="progress-bar-fill" style="width: {pct}%;"></div>
    </div>
    """
    if label:
        html += f'<span class="caption">{label}</span>'
    return html


def candidate_row(candidate_id: str, recommendation: str | None, summary: str, score: float | None = None) -> str:
    """Return HTML for a candidate list row.

    This is a row, not a card — hairline divider between rows,
    no shadow, no large rounded corners.
    """
    score_html = ""
    if score is not None:
        # Score is 0-100; display as integer with /100 suffix
        score_html = f'<span style="font-size:13px;color:var(--muted-2);margin-left:16px;">{score:.0f}/100</span>'
    pill = status_pill(recommendation)
    return f"""
    <div class="candidate-row" onclick="this.querySelector('a').click()">
        <span class="candidate-row-name">{candidate_id}</span>
        {pill}
        <span class="candidate-row-summary">{summary}</span>
        {score_html}
        <span class="candidate-row-chevron">›</span>
    </div>
    """


def alert_warn(message: str) -> str:
    """Return HTML for a warning alert callout."""
    return f'<div class="alert alert-warn">{message}</div>'


def evidence_list(items: list[str]) -> str:
    """Return HTML for an evidence list."""
    items_html = "\n".join(
        f'<div class="evidence-item">{item}</div>' for item in items
    )
    return f'<div class="evidence-list">{items_html}</div>'


def expandable_section(title: str, content: str, expanded: bool = False) -> str:
    """Return HTML for an expandable section.

    Uses a simple checkbox hack for pure CSS toggle behavior.
    """
    checked = "checked" if expanded else ""
    return f"""
    <div class="expandable-section" style="border-bottom:1px solid var(--line);">
        <input type="checkbox" id="expand-{title.lower().replace(' ', '-')}" {checked}
               style="display:none;">
        <label for="expand-{title.lower().replace(' ', '-')}" class="expandable-header">
            <span>{title}</span>
            <span style="font-size:13px;color:var(--muted-2);transition:transform 120ms ease-out;">▸</span>
        </label>
        <div class="expandable-content" style="display:{'block' if expanded else 'none'};">
            {content}
        </div>
    </div>
    <script>
        document.getElementById('expand-{title.lower().replace(' ', '-')}')
            .addEventListener('change', function() {{
                var content = this.nextElementSibling.nextElementSibling;
                var arrow = this.nextElementSibling.querySelector('span:last-child');
                content.style.display = this.checked ? 'block' : 'none';
                arrow.style.transform = this.checked ? 'rotate(90deg)' : 'rotate(0deg)';
            }});
    </script>
    """