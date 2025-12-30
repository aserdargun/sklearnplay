"""Custom CSS styles for glassmorphism UI.

This module provides centralized CSS injection for a modern glassmorphism
design with light mode colors.
"""

import streamlit as st


def get_glassmorphism_css() -> str:
    """Return the main glassmorphism CSS stylesheet."""
    return """
    <style>
    /* ===== ROOT VARIABLES ===== */
    :root {
        --primary: #6366f1;
        --primary-light: #818cf8;
        --primary-dark: #4f46e5;
        --bg-primary: #f8fafc;
        --bg-secondary: #f1f5f9;
        --bg-glass: rgba(255, 255, 255, 0.7);
        --border-glass: rgba(255, 255, 255, 0.3);
        --shadow-soft: 0 8px 32px rgba(0, 0, 0, 0.08);
        --shadow-hover: 0 12px 40px rgba(0, 0, 0, 0.12);
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --text-muted: #94a3b8;
    }

    /* ===== BASE STYLES ===== */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* ===== GLASSMORPHISM CARD BASE ===== */
    .glass-card {
        background: var(--bg-glass);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid var(--border-glass);
        box-shadow: var(--shadow-soft);
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-hover);
    }

    /* ===== SIDEBAR STYLING ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(241,245,249,0.98) 100%);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }

    section[data-testid="stSidebar"] > div:first-child {
        background: transparent;
        padding-top: 1rem;
    }

    /* ===== HERO SECTION ===== */
    .hero-container {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(139, 92, 246, 0.08) 100%);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        padding: 2.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }

    .hero-container h1 {
        color: var(--text-primary);
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
    }

    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 1.2rem;
        margin: 0;
    }

    .hero-logo {
        max-width: 280px;
        height: auto;
        margin-bottom: 1rem;
    }

    /* ===== NAVIGATION CARDS ===== */
    .nav-card {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
        padding: 1.25rem;
        text-align: center;
        transition: all 0.2s ease;
        height: 100%;
    }

    .nav-card:hover {
        background: rgba(255, 255, 255, 0.8);
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
    }

    .nav-card h4 {
        color: var(--text-primary);
        margin: 0.5rem 0;
        font-weight: 600;
    }

    .nav-card p {
        color: var(--text-secondary);
        font-size: 0.875rem;
        margin: 0;
    }

    /* ===== ICON BADGE ===== */
    .icon-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
        border-radius: 12px;
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }

    .icon-badge-sm {
        width: 36px;
        height: 36px;
        font-size: 1.2rem;
        border-radius: 8px;
    }

    /* ===== FEATURE CARDS ===== */
    .feature-card {
        background: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 1rem;
        text-align: center;
        height: 100%;
    }

    .feature-card .icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }

    .feature-card h5 {
        color: var(--text-primary);
        font-weight: 600;
        margin: 0.25rem 0;
        font-size: 0.95rem;
    }

    .feature-card p {
        color: var(--text-muted);
        font-size: 0.8rem;
        margin: 0;
    }

    /* ===== LEVEL SELECTOR STYLING ===== */
    .level-container {
        background: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .level-container h4 {
        color: var(--text-primary);
        font-weight: 600;
        margin-bottom: 0.75rem;
        font-size: 0.95rem;
    }

    /* ===== METRIC CARDS ===== */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }

    div[data-testid="stMetric"] label {
        color: var(--text-secondary) !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-weight: 600;
    }

    /* ===== TABS STYLING ===== */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        border-radius: 12px;
        padding: 0.5rem;
        gap: 0.25rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        background: transparent;
        color: var(--text-secondary);
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.1);
        color: var(--primary);
    }

    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
        border: none;
        border-radius: 10px;
        color: white;
        font-weight: 500;
        padding: 0.5rem 1.5rem;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Secondary buttons */
    .stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(6px);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: var(--primary);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }

    .stButton > button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.9);
        border-color: var(--primary);
    }

    /* ===== EXPANDERS ===== */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        font-weight: 500;
    }

    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 0 0 10px 10px;
    }

    /* ===== SELECTBOX / MULTISELECT ===== */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(6px);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.4);
    }

    /* ===== TEXT INPUT ===== */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(6px);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.4);
    }

    /* ===== DATAFRAMES ===== */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        overflow: hidden;
    }

    /* ===== DIVIDERS ===== */
    .glass-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.2), transparent);
        margin: 1rem 0;
        border: none;
    }

    /* ===== SECTION HEADERS ===== */
    .section-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }

    .section-header h3 {
        color: var(--text-primary);
        font-weight: 600;
        margin: 0;
    }

    /* ===== DATASET ROW ===== */
    .dataset-row {
        background: rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(4px);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.2s ease;
    }

    .dataset-row:hover {
        background: rgba(255, 255, 255, 0.6);
        transform: translateX(4px);
    }

    /* ===== LOGO STYLING ===== */
    .logo-container {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
    }

    .logo-container img {
        filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.1));
    }

    /* ===== TYPOGRAPHY ===== */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary);
    }

    h1 {
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    h2 {
        font-weight: 600;
    }

    h3 {
        font-weight: 600;
    }

    p {
        color: var(--text-secondary);
    }

    /* ===== BROWSER FALLBACKS ===== */
    @supports not (backdrop-filter: blur(10px)) {
        .glass-card,
        .nav-card,
        .feature-card,
        .hero-container,
        .level-container {
            background: rgba(255, 255, 255, 0.95);
        }

        section[data-testid="stSidebar"] {
            background: rgba(248, 250, 252, 0.98);
        }
    }

    /* ===== ANIMATIONS ===== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .animate-fade-in {
        animation: fadeIn 0.3s ease-out;
    }
    </style>
    """


def inject_styles() -> None:
    """Inject the glassmorphism CSS into the Streamlit app.

    Call this function once at the start of your app, right after st.set_page_config().
    """
    st.markdown(get_glassmorphism_css(), unsafe_allow_html=True)


def hero_section(title: str, subtitle: str, logo: str | None = None) -> str:
    """Create a hero section with glassmorphism styling.

    Args:
        title: Main title text
        subtitle: Subtitle/description text
        logo: Optional path to logo image

    Returns:
        HTML string for the hero section
    """
    logo_html = f'<img src="app/static/{logo}" alt="Logo" class="hero-logo">' if logo else ""
    return f'''
    <div class="hero-container animate-fade-in">
        {logo_html}
        <h1>{title}</h1>
        <p class="hero-subtitle">{subtitle}</p>
    </div>
    '''


def glass_card(content: str, extra_class: str = "") -> str:
    """Wrap content in a glassmorphism card div.

    Args:
        content: HTML content to wrap
        extra_class: Additional CSS classes to add

    Returns:
        HTML string for the glass card
    """
    classes = f"glass-card {extra_class}".strip()
    return f'<div class="{classes}">{content}</div>'


def nav_card(icon: str, title: str, description: str) -> str:
    """Create a navigation card with glassmorphism styling.

    Args:
        icon: Emoji or icon character
        title: Card title
        description: Card description

    Returns:
        HTML string for the navigation card
    """
    return f'''
    <div class="nav-card">
        <div class="icon-badge">{icon}</div>
        <h4>{title}</h4>
        <p>{description}</p>
    </div>
    '''


def feature_card(icon: str, title: str, description: str) -> str:
    """Create a small feature card.

    Args:
        icon: Emoji or icon character
        title: Feature title
        description: Feature description

    Returns:
        HTML string for the feature card
    """
    return f'''
    <div class="feature-card">
        <div class="icon">{icon}</div>
        <h5>{title}</h5>
        <p>{description}</p>
    </div>
    '''


def section_header(title: str, icon: str = "") -> str:
    """Create a styled section header.

    Args:
        title: Section title
        icon: Optional emoji/icon

    Returns:
        HTML string for the section header
    """
    icon_html = f'<div class="icon-badge-sm" style="margin-right: 0.75rem;">{icon}</div>' if icon else ""
    return f'''
    <div class="section-header">
        {icon_html}
        <h3>{title}</h3>
    </div>
    '''


def glass_divider() -> str:
    """Create a styled divider.

    Returns:
        HTML string for the divider
    """
    return '<div class="glass-divider"></div>'


def dataset_row(name: str, description: str, samples: int, task_type: str) -> str:
    """Create a styled dataset row.

    Args:
        name: Dataset name
        description: Dataset description
        samples: Number of samples
        task_type: Task type (classification, regression, etc.)

    Returns:
        HTML string for the dataset row
    """
    return f'''
    <div class="dataset-row">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong style="color: var(--text-primary);">{name}</strong>
                <span style="color: var(--text-muted); margin-left: 0.5rem;">{description[:50]}...</span>
            </div>
            <div style="display: flex; gap: 1rem;">
                <span style="color: var(--text-secondary); font-size: 0.875rem;">{samples} samples</span>
                <span style="color: var(--primary); font-size: 0.875rem;">{task_type}</span>
            </div>
        </div>
    </div>
    '''
