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
    """Return the premium, two-column tech-editorial UI layout overrides for ScoutAI."""
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;500;600;700&family=Space+Mono&display=swap');

    /* ── Core Canvas Baseline & Faint Tech Blueprint Grid ──────────────── */
    html, body, .stApp {{
        background-color: #F8F8F6 !important;
        color: #0A0A0A !important;
        font-family: 'Open Sans', -apple-system, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }}

    .stApp {{
        background-image: 
            linear-gradient(to right, rgba(0,0,0,0.015) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(0,0,0,0.015) 1px, transparent 1px) !important;
        background-size: 32px 32px !important;
    }}

    /* ── Asymmetric Column Layout Engine ───────────────────────────────── */
    [data-testid="stAppViewBlockContainer"], .block-container {{
        max-width: 1200px !important;
        margin: 0 auto !important;
        padding: 80px 48px !important;
    }}

    /* Target the horizontal column block container */
    [data-testid="stHorizontalBlock"] {{
        display: flex !important;
        flex-direction: row !important;
        gap: 96px !important; 
        align-items: flex-start !important;
        margin-top: 48px !important;
    }}

    /* Balance both sides evenly */
    [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {{
        width: 50% !important;
        flex: 1 1 0% !important;
    }}

    /* Make left column text cleanly sticky */
    [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child {{
        position: sticky !important;
        top: 140px !important;
    }}

    /* ── Absolute Branding Placement ─────────────────────────────────────── */
    .brand-logo {{
        position: absolute;
        top: 40px;
        left: 0px;
        font-family: 'Space Mono', monospace;
        font-size: 22px !important;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #0A0A0A;
    }}

    /* ── Typography Scale Correction ──────────────────────────────────────── */
    h1 {{
        font-family: 'Open Sans', sans-serif !important;
        font-size: 52px !important;
        font-weight: 700 !important;
        letter-spacing: -0.04em !important;
        line-height: 1.05 !important;
        color: #0A0A0A !important;
        margin-top: 24px !important;
        margin-bottom: 16px !important;
    }}

    .eyebrow {{
        font-family: 'Space Mono', monospace !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        color: #6B6B6B !important;
        margin-bottom: 8px !important;
    }}

    [data-testid="stWidgetLabel"] p {{
        font-family: 'Space Mono', monospace !important;
        font-size: 20px !important; 
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #0A0A0A !important;
        margin-bottom: 12px !important;
    }}

    /* ── Sharp Input Workspace Fields ─────────────────────────────────────── */
    .stTextInput input, .stTextArea textarea {{
        background-color: #FFFFFF !important;
        border: 1px solid #E4E4E0 !important;
        border-radius: 0px !important;
        padding: 16px !important;
        font-size: 14px !important;
        transition: border-color 150ms ease !important;
    }}
    
    .stTextInput input:focus, .stTextArea textarea:focus {{
        border-color: #0A0A0A !important;
    }}

    /* ── Custom File Dropzone UI ─────────────────────────────────────────── */
    [data-testid="stFileUploader"] > section {{
        background-color: #FFFFFF !important;
        border: 1px dashed #D4D4CE !important;
        border-radius: 0px !important;
        padding: 24px !important;
        display: flex !important;
        align-items: center !important;
        gap: 16px !important;
    }}

    [data-testid="stFileUploader"] [data-testid="stCaptionContainer"] p {{
        font-family: 'Space Mono', monospace !important;
        font-size: 14px !important; /* Bumped up for clean technical scanning */
        color: #6B6B6B !important;
    }}

    [data-testid="stFileUploader"] button[data-testid="baseButton-secondary"] {{
        background-color: #0A0A0A !important;
        color: #FFFFFF !important;
        border: 1px solid #0A0A0A !important;
        border-radius: 0px !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 11px !important;
        text-transform: uppercase !important;
        padding: 8px 16px !important;
        height: auto !important;
    }}

    /* ── Clean Call-To-Action Block Button ────────────────────────────────── */
    .stButton button {{
        background-color: #0A0A0A !important;
        color: #FFFFFF !important;
        border: 1px solid #0A0A0A !important;
        border-radius: 0px !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        padding: 16px 40px !important;
        width: 100% !important;
        transition: all 150ms ease !important;
        box-shadow: 4px 4px 0px rgba(0,0,0,0.05) !important;
    }}

    .stButton button:hover {{
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        transform: translate(1px, 1px) !important;
        box-shadow: 2px 2px 0px rgba(0,0,0,0.02) !important;
    }}

    /* Hide base boilerplate chrome elements */
    header[data-testid="stHeader"], [data-testid="stDecoration"] {{
        display: none !important;
    }}
    </style>
    """