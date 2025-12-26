"""scikit-learn Playground - Main Streamlit Application.

An interactive learning platform for scikit-learn, structured to follow
the official User Guide with progressive complexity levels.
"""

import streamlit as st

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="scikit-learn Playground",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


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

    # Sidebar
    with st.sidebar:
        st.image("https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png", width=200)
        st.title("sklearn Playground")
        st.markdown("---")

        # Level selector
        st.subheader("Experience Level")
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

        st.markdown("---")

        # Quick links
        st.subheader("Resources")
        st.markdown("""
        - [sklearn User Guide](https://scikit-learn.org/stable/user_guide.html)
        - [sklearn API Reference](https://scikit-learn.org/stable/api/index.html)
        - [sklearn Examples](https://scikit-learn.org/stable/auto_examples/index.html)
        """)

    # Main content
    st.title("Welcome to scikit-learn Playground 🧪")

    st.markdown("""
    An interactive platform to learn and experiment with scikit-learn,
    structured around the official User Guide.

    ### Getting Started

    Use the sidebar navigation to explore different topics:

    1. **Supervised Learning** - Classification and regression algorithms
    2. **Unsupervised Learning** - Clustering, dimensionality reduction
    3. **Model Selection** - Cross-validation, hyperparameter tuning
    4. **And more...** - Following the sklearn User Guide structure

    ### Features

    - 📊 **Toy Datasets** - Pre-loaded datasets by domain (power, retail, finance, healthcare)
    - 📤 **CSV Upload** - Use your own data
    - 🎛️ **Interactive Controls** - Adjust parameters and see results
    - 📈 **Visualizations** - Learning curves, metrics, feature importance
    - 💾 **Export** - Download trained models and code snippets
    """)

    # Quick start section
    st.markdown("---")
    st.subheader("Quick Start")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🎯 Classification")
        st.markdown("Predict categories from features")
        if st.button("Start Classification", key="quick_clf"):
            st.switch_page("pages/01_supervised_learning.py")

    with col2:
        st.markdown("#### 📉 Regression")
        st.markdown("Predict continuous values")
        if st.button("Start Regression", key="quick_reg"):
            st.switch_page("pages/01_supervised_learning.py")

    with col3:
        st.markdown("#### 🔍 API Explorer")
        st.markdown("Search sklearn classes and functions")
        if st.button("Explore API", key="quick_api"):
            st.switch_page("pages/90_api_explorer.py")

    # Dataset overview
    st.markdown("---")
    st.subheader("Available Datasets")

    from skplay.core.datasets import DatasetRegistry

    domains = ["general", "power", "retail", "finance", "healthcare"]
    tabs = st.tabs([d.title() for d in domains])

    for tab, domain in zip(tabs, domains):
        with tab:
            datasets = DatasetRegistry.list_by_domain(domain)
            if datasets:
                for name in datasets:
                    card = DatasetRegistry.get_card(name)
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.markdown(f"**{name}** - {card.description[:60]}...")
                        with col2:
                            st.caption(f"📊 {card.n_samples} samples")
                        with col3:
                            st.caption(f"🎯 {card.task_type}")
            else:
                st.info(f"No datasets for {domain} domain")


if __name__ == "__main__":
    main()
