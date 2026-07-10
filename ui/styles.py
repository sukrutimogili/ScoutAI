"""
Design tokens and CSS injection for ScoutAI Streamlit UI.

Every visual decision derives from the token table in §1 of the design spec.
No color, weight, or radius is introduced that isn't listed here.
"""

from __future__ import annotations

# ── Design Tokens (§1) ────────────────────────────────────────────────────────

TOKENS = {
    "--ink": "#0A0A0A",
    "--paper": "#FFFFFF",
    "--paper-dim": "#FAFAFA",
    "--line": "#E4E4E4",
    "--line-strong": "#0A0A0A",
    "--muted": "#6B6B6B",
    "--muted-2": "#9B9B9B",
    "--accent": "#0A0A0A",
    "--signal-good-bg": "#E9F5EC",
    "--signal-warn-bg": "#FBF3DC",
    "--signal-bad-bg": "#FBEAE9",
    "--signal-neutral-bg": "#EFEFEF",
    "--radius": "2px",
    "--max-width": "960px",
    "--font-family": "Open Sans, system-ui, sans-serif",
}

# ── Spacing scale (§1) ────────────────────────────────────────────────────────

SPACING = {4, 8, 12, 16, 24, 32, 48, 64, 96}


def inject_css() -> str:
    """Return the full CSS <style> block to inject into the Streamlit app.

    This is called once at the top of app.py via st.markdown(..., unsafe_allow_html=True).
    It overrides Streamlit's default theme to match the ScoutAI design tokens.
    """
    return f"""
    @import url('https://fonts.googleapis.com/css2?family=Open Sans:wght@400;700&display=swap');

    * {{
        font-family: '{TOKENS['--font-family']}' !important;
    }}

    html, body, .stApp {{
        background-color: {TOKENS['--paper']};
        color: {TOKENS['--ink']};
        font-family: '{TOKENS['--font-family']}';
        font-weight: 400;
        font-size: 15px;
        line-height: 1.6;
    }}

    /* ── Masthead ─────────────────────────────────────────────────────────── */
    .masthead {{
        height: 56px;
        border-bottom: 1px solid {TOKENS['--line']};
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 32px;
        background: {TOKENS['--paper']};
        position: sticky;
        top: 0;
        z-index: 100;
    }}
    .masthead-wordmark {{
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        line-height: 1;
        color: {TOKENS['--ink']};
    }}
    .masthead-right {{
        font-size: 13px;
        color: {TOKENS['--muted']};
    }}
    .masthead-right a {{
        color: {TOKENS['--muted']};
        text-decoration: underline;
        cursor: pointer;
        margin-left: 16px;
    }}
    .masthead-right a:hover {{
        color: {TOKENS['--ink']};
    }}

    /* ── Content container ────────────────────────────────────────────────── */
    .content {{
        max-width: {TOKENS['--max-width']};
        margin: 0 auto;
        padding: 48px 32px;
    }}

    /* ── Typography ───────────────────────────────────────────────────────── */
    h1 {{
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.005em;
        line-height: 1.2;
        margin: 0 0 32px 0;
        color: {TOKENS['--ink']};
    }}
    h2 {{
        font-size: 18px;
        font-weight: 700;
        line-height: 1.3;
        margin: 0 0 16px 0;
        color: {TOKENS['--ink']};
    }}
    .eyebrow {{
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        line-height: 1;
        color: {TOKENS['--muted']};
        margin-bottom: 8px;
    }}
    .display-number {{
        font-size: 64px;
        font-weight: 700;
        letter-spacing: -0.01em;
        line-height: 1.05;
        color: {TOKENS['--ink']};
        margin: 0;
    }}
    .caption {{
        font-size: 13px;
        font-weight: 400;
        line-height: 1.5;
        color: {TOKENS['--muted']};
    }}
    .caption-2 {{
        font-size: 13px;
        font-weight: 400;
        color: {TOKENS['--muted-2']};
    }}

    /* ── Status pills ─────────────────────────────────────────────────────── */
    .status-pill {{
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: {TOKENS['--radius']};
        font-size: 13px;
        font-weight: 700;
        color: {TOKENS['--ink']};
        line-height: 1.4;
        white-space: nowrap;
    }}
    .status-pill.strong_interview {{ background: {TOKENS['--signal-good-bg']}; }}
    .status-pill.interview {{ background: {TOKENS['--signal-neutral-bg']}; }}
    .status-pill.reject {{ background: {TOKENS['--signal-bad-bg']}; }}
    .status-pill.hold {{ background: {TOKENS['--signal-warn-bg']}; }}

    /* ── Buttons ──────────────────────────────────────────────────────────── */
    .btn-primary {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 8px 24px;
        background: {TOKENS['--ink']};
        color: {TOKENS['--paper']};
        border: 1px solid {TOKENS['--ink']};
        border-radius: {TOKENS['--radius']};
        font-size: 15px;
        font-weight: 700;
        line-height: 1.4;
        cursor: pointer;
        transition: opacity 120ms ease-out;
        text-decoration: none;
    }}
    .btn-primary:hover {{
        opacity: 0.85;
    }}
    .btn-primary:disabled {{
        background: {TOKENS['--muted-2']};
        border-color: {TOKENS['--muted-2']};
        cursor: not-allowed;
        opacity: 1;
    }}
    .btn-secondary {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 8px 24px;
        background: transparent;
        color: {TOKENS['--ink']};
        border: 1px solid {TOKENS['--line']};
        border-radius: {TOKENS['--radius']};
        font-size: 15px;
        font-weight: 400;
        line-height: 1.4;
        cursor: pointer;
        transition: border-color 120ms ease-out;
        text-decoration: none;
    }}
    .btn-secondary:hover {{
        border-color: {TOKENS['--ink']};
    }}
    .btn-text {{
        display: inline-flex;
        align-items: center;
        padding: 4px 8px;
        background: transparent;
        color: {TOKENS['--muted']};
        border: none;
        border-radius: {TOKENS['--radius']};
        font-size: 13px;
        font-weight: 400;
        cursor: pointer;
        text-decoration: underline;
        transition: color 120ms ease-out;
    }}
    .btn-text:hover {{
        color: {TOKENS['--ink']};
    }}

    /* ── Cards / sections ─────────────────────────────────────────────────── */
    .card {{
        border: 1px solid {TOKENS['--line']};
        border-radius: {TOKENS['--radius']};
        padding: 24px;
        margin-bottom: 16px;
        background: {TOKENS['--paper']};
    }}
    .card-dim {{
        background: {TOKENS['--paper-dim']};
    }}

    /* ── Hairline dividers ────────────────────────────────────────────────── */
    hr {{
        border: none;
        border-top: 1px solid {TOKENS['--line']};
        margin: 16px 0;
    }}

    /* ── Candidate row ────────────────────────────────────────────────────── */
    .candidate-row {{
        display: flex;
        align-items: center;
        padding: 16px 0;
        border-bottom: 1px solid {TOKENS['--line']};
        cursor: pointer;
        transition: background 120ms ease-out;
    }}
    .candidate-row:hover {{
        background: {TOKENS['--paper-dim']};
    }}
    .candidate-row:last-child {{
        border-bottom: none;
    }}
    .candidate-row-name {{
        flex: 1;
        font-size: 15px;
        font-weight: 700;
        color: {TOKENS['--ink']};
    }}
    .candidate-row-summary {{
        flex: 2;
        font-size: 13px;
        color: {TOKENS['--muted']};
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        padding: 0 16px;
    }}
    .candidate-row-chevron {{
        color: {TOKENS['--muted-2']};
        font-size: 15px;
        margin-left: 8px;
    }}

    /* ── Stat blocks ──────────────────────────────────────────────────────── */
    .stat-block {{
        display: inline-flex;
        flex-direction: column;
        align-items: flex-start;
        padding: 16px 24px;
        border: 1px solid {TOKENS['--line']};
        border-radius: {TOKENS['--radius']};
        background: {TOKENS['--paper']};
    }}
    .stat-block-number {{
        font-size: 28px;
        font-weight: 700;
        line-height: 1.2;
        color: {TOKENS['--ink']};
    }}
    .stat-block-label {{
        font-size: 13px;
        font-weight: 400;
        color: {TOKENS['--muted']};
        margin-top: 4px;
    }}

    /* ── Progress bar ─────────────────────────────────────────────────────── */
    .progress-bar {{
        height: 4px;
        background: {TOKENS['--line']};
        border-radius: {TOKENS['--radius']};
        overflow: hidden;
        margin: 8px 0;
    }}
    .progress-bar-fill {{
        height: 100%;
        background: {TOKENS['--ink']};
        border-radius: {TOKENS['--radius']};
        transition: width 180ms ease-out;
    }}

    /* ── Step list (Processing screen) ────────────────────────────────────── */
    .step-list {{
        list-style: none;
        padding: 0;
        margin: 0;
    }}
    .step-item {{
        display: flex;
        align-items: center;
        padding: 12px 0;
        border-bottom: 1px solid {TOKENS['--line']};
    }}
    .step-item:last-child {{
        border-bottom: none;
    }}
    .step-index {{
        width: 32px;
        font-size: 15px;
        font-weight: 700;
        color: {TOKENS['--muted-2']};
        flex-shrink: 0;
    }}
    .step-label {{
        flex: 1;
        font-size: 15px;
        font-weight: 400;
        color: {TOKENS['--muted']};
    }}
    .step-label.active {{
        font-weight: 700;
        color: {TOKENS['--ink']};
    }}
    .step-label.done {{
        color: {TOKENS['--ink']};
    }}
    .step-indicator {{
        width: 20px;
        height: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        font-size: 13px;
        color: {TOKENS['--muted-2']};
    }}
    .step-indicator.active {{
        color: {TOKENS['--ink']};
    }}
    .step-indicator.done {{
        color: {TOKENS['--ink']};
    }}

    /* ── Alert callout ────────────────────────────────────────────────────── */
    .alert {{
        border: 1px solid {TOKENS['--line']};
        border-radius: {TOKENS['--radius']};
        padding: 12px 16px;
        margin-bottom: 16px;
        font-size: 13px;
        line-height: 1.5;
    }}
    .alert-warn {{
        background: {TOKENS['--signal-warn-bg']};
        border-color: transparent;
    }}

    /* ── Expandable section ───────────────────────────────────────────────── */
    .expandable-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 0;
        cursor: pointer;
        border-bottom: 1px solid {TOKENS['--line']};
        font-size: 15px;
        font-weight: 700;
        color: {TOKENS['--ink']};
        transition: color 120ms ease-out;
    }}
    .expandable-header:hover {{
        color: {TOKENS['--muted']};
    }}
    .expandable-content {{
        padding: 16px 0;
        font-size: 15px;
        line-height: 1.6;
        color: {TOKENS['--ink']};
    }}

    /* ── Evidence list ────────────────────────────────────────────────────── */
    .evidence-item {{
        padding: 8px 0;
        border-bottom: 1px solid {TOKENS['--line']};
        font-size: 13px;
        line-height: 1.5;
        color: {TOKENS['--muted']};
    }}
    .evidence-item:last-child {{
        border-bottom: none;
    }}

    /* ── Comparison grid ───────────────────────────────────────────────────── */
    .comparison-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 24px;
    }}
    .comparison-col {{
        border: 1px solid {TOKENS['--line']};
        border-radius: {TOKENS['--radius']};
        padding: 24px;
    }}

    /* ── Export buttons ───────────────────────────────────────────────────── */
    .export-row {{
        display: flex;
        gap: 16px;
        margin-top: 24px;
    }}

    /* ── Override Streamlit defaults ──────────────────────────────────────── */
    .stButton > button {{
        border-radius: {TOKENS['--radius']} !important;
        font-family: '{TOKENS['--font-family']}' !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        padding: 8px 24px !important;
        transition: opacity 120ms ease-out !important;
    }}
    .stButton > button[kind="primary"] {{
        background: {TOKENS['--ink']} !important;
        color: {TOKENS['--paper']} !important;
        border: 1px solid {TOKENS['--ink']} !important;
    }}
    .stButton > button[kind="secondary"] {{
        background: transparent !important;
        color: {TOKENS['--ink']} !important;
        border: 1px solid {TOKENS['--line']} !important;
    }}
    .stTextInput > div > div > input {{
        border-radius: {TOKENS['--radius']} !important;
        border: 1px solid {TOKENS['--line']} !important;
        font-family: '{TOKENS['--font-family']}' !important;
        font-size: 15px !important;
    }}
    .stFileUploader > div {{
        border-radius: {TOKENS['--radius']} !important;
        border: 1px dashed {TOKENS['--line']} !important;
    }}
    .stSelectbox > div > div {{
        border-radius: {TOKENS['--radius']} !important;
        border: 1px solid {TOKENS['--line']} !important;
    }}

    /* ── Responsive: mobile ───────────────────────────────────────────────── */
    @media (max-width: 768px) {{
        .content {{
            padding: 24px 20px;
        }}
        .masthead {{
            padding: 0 20px;
        }}
        .candidate-row {{
            flex-wrap: wrap;
        }}
        .candidate-row-summary {{
            flex-basis: 100%;
            padding: 8px 0 0 0;
        }}
        .comparison-grid {{
            grid-template-columns: 1fr;
        }}
        .export-row {{
            flex-direction: column;
        }}
        .stat-block {{
            width: 100%;
        }}
    }}

    /* ── prefers-reduced-motion ────────────────────────────────────────────── */
    @media (prefers-reduced-motion: reduce) {{
        * {{
            transition-duration: 0ms !important;
            animation-duration: 0ms !important;
        }}
        .btn-primary, .btn-secondary, .candidate-row {{
            transition: none !important;
        }}
        .progress-bar-fill {{
            transition: none !important;
        }}
    }}
    """