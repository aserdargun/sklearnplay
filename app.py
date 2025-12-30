"""scikit-learn Playground - Main Streamlit Application.

An interactive learning platform for scikit-learn, structured to follow
the official User Guide with progressive complexity levels.
"""

import streamlit as st

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="scikit-learn Playground",
    page_icon="images/icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject glassmorphism styles immediately after page config
from skplay.ui.styles import (  # noqa: E402
    dataset_row,
    feature_card,
    glass_divider,
    hero_section,
    inject_styles,
    nav_card,
)

inject_styles()


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "user_level": "beginner",
        "selected_dataset": None,
        "selected_task": None,
        "trained_model": None,
        "training_results": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    """Main application entry point."""
    init_session_state()

    # Sidebar with modern styling
    with st.sidebar:
        # Experience Level selector FIRST - so user adjusts it first
        st.markdown('<div class="level-container">', unsafe_allow_html=True)
        st.markdown('<h4>Experience Level</h4>', unsafe_allow_html=True)

        level_options = {
            "beginner": "🌱 Beginner",
            "intermediate": "🌿 Intermediate",
            "advanced": "🌳 Advanced",
        }

        current_level = st.session_state.user_level
        selected_level = st.radio(
            "Choose your level",
            options=list(level_options.keys()),
            format_func=lambda x: level_options[x],
            index=list(level_options.keys()).index(current_level),
            label_visibility="collapsed",
        )

        if selected_level != current_level:
            st.session_state.user_level = selected_level
            st.rerun()

        level_descriptions = {
            "beginner": "Guided flow with essential controls",
            "intermediate": "More tuning options and validation",
            "advanced": "Full control over all parameters",
        }
        st.caption(level_descriptions[selected_level])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(glass_divider(), unsafe_allow_html=True)

        # Logo with styled container
        st.image("images/logo.png", width="stretch")
        st.markdown(glass_divider(), unsafe_allow_html=True)

        # Quick links with styled container
        st.markdown('<div class="level-container">', unsafe_allow_html=True)
        st.markdown('<h4>Resources</h4>', unsafe_allow_html=True)
        st.markdown(
            """
        - [User Guide](https://scikit-learn.org/stable/user_guide.html)
        - [API Reference](https://scikit-learn.org/stable/api/index.html)
        - [Examples](https://scikit-learn.org/stable/auto_examples/index.html)
        """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Main content - Hero section
    st.markdown(
        hero_section(
            "scikit-learn Playground with Streamlit",
            "Interactive Machine Learning Education Platform Developed with AI Pair Programming",
            logo="logo.png",
        ),
        unsafe_allow_html=True,
    )

    # Open source and GitHub section
    st.markdown(
        """
        <div style="text-align: center; margin: -1rem 0 1.5rem 0;">
            <a href="https://github.com/aserdargun/sklearnplay" target="_blank"
               style="display: inline-flex; align-items: center; gap: 0.5rem;
                      background: linear-gradient(135deg, rgba(36, 41, 46, 0.9), rgba(22, 27, 34, 0.95));
                      color: #fff; padding: 0.6rem 1.2rem; border-radius: 8px;
                      text-decoration: none; font-weight: 500; font-size: 0.95rem;
                      border: 1px solid rgba(255, 255, 255, 0.1);
                      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                      transition: all 0.2s ease;">
                <svg height="20" width="20" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                    0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52
                    -.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2
                    -3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82
                    .64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08
                    2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01
                    1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
                </svg>
                Open Source on GitHub
            </a>
            <p style="margin-top: 0.75rem; color: rgba(255, 255, 255, 0.7); font-size: 0.85rem;">
                AI pair programmed with Claude Code
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Features row
    col1, col2, col3, col4 = st.columns(4)
    features = [
        ("📊", "Toy Datasets", "Pre-loaded domain datasets"),
        ("📤", "CSV Upload", "Use your own data"),
        ("🎛️", "Interactive", "Adjust and visualize"),
        ("💾", "Export", "Download models & code"),
    ]

    for col, (icon, title, desc) in zip([col1, col2, col3, col4], features, strict=True):
        with col:
            st.markdown(feature_card(icon, title, desc), unsafe_allow_html=True)

    # Quick start section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header"><h3>🚀 Quick Start</h3></div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            nav_card("🎯", "Classification", "Predict categories from features"),
            unsafe_allow_html=True,
        )
        if st.button("Start Classification", key="quick_clf", width="stretch"):
            st.switch_page("pages/01_supervised_learning.py")

    with col2:
        st.markdown(
            nav_card("📉", "Regression", "Predict continuous values"),
            unsafe_allow_html=True,
        )
        if st.button("Start Regression", key="quick_reg", width="stretch"):
            st.switch_page("pages/01_supervised_learning.py")

    with col3:
        st.markdown(
            nav_card("🔍", "API Explorer", "Search sklearn classes & functions"),
            unsafe_allow_html=True,
        )
        if st.button("Explore API", key="quick_api", width="stretch"):
            st.switch_page("pages/90_api_explorer.py")

    # Dataset overview
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header"><h3>📊 Available Datasets</h3></div>',
        unsafe_allow_html=True,
    )

    from skplay.core.datasets import DatasetRegistry

    domains = ["general", "power", "retail", "finance", "healthcare"]
    tabs = st.tabs([f"  {d.title()}  " for d in domains])

    for tab, domain in zip(tabs, domains, strict=True):
        with tab:
            datasets = DatasetRegistry.list_by_domain(domain)
            if datasets:
                for name in datasets:
                    card = DatasetRegistry.get_card(name)
                    st.markdown(
                        dataset_row(
                            name=name.replace("_", " ").title(),
                            description=card.description,
                            samples=card.n_samples,
                            task_type=card.task_type.replace("_", " ").title(),
                        ),
                        unsafe_allow_html=True,
                    )
            else:
                st.info(f"No datasets for {domain} domain")

    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center; color: var(--text-muted); font-size: 0.875rem;">
            Built with Streamlit and scikit-learn
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
