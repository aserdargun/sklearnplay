"""API Explorer.

Search and explore the scikit-learn API through introspection.
"""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="API Explorer", page_icon="images/icon.png", layout="wide")

from skplay.core.api_explorer import (
    generate_example_snippet,
    get_api_entry,
    get_explorer,
    search_api,
)
from skplay.ui.level import level_selector


def main():
    st.title("🔎 API Explorer")

    with st.sidebar:
        level_selector()

    st.markdown("""
    Search and explore scikit-learn classes, functions, and estimators.

    [📚 sklearn API Reference](https://scikit-learn.org/stable/api/index.html)
    """)

    # Search interface
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        query = st.text_input(
            "Search",
            placeholder="Enter class or function name...",
            key="api_search",
        )

    with col2:
        kind_filter = st.selectbox(
            "Type",
            options=[None, "class", "function", "display"],
            format_func=lambda x: "All" if x is None else x.title(),
            key="api_kind",
        )

    with col3:
        type_filter = st.selectbox(
            "Estimator Type",
            options=[
                None,
                "classifier",
                "regressor",
                "clusterer",
                "transformer",
                "outlier_detector",
            ],
            format_func=lambda x: "All" if x is None else x.title(),
            key="api_type",
        )

    st.markdown("---")

    # Initialize explorer
    explorer = get_explorer()

    if query:
        # Search results
        results = search_api(
            query,
            kind=kind_filter,
            estimator_type=type_filter,
            limit=30,
        )

        if results:
            st.subheader(f"Found {len(results)} results")

            for entry in results:
                with st.expander(f"**{entry.name}** ({entry.kind}) - {entry.module}"):
                    display_entry(entry)
        else:
            st.info("No results found. Try a different search term.")

    else:
        # Browse by category
        st.subheader("Browse by Category")

        categories = {
            "Classifiers": "classifier",
            "Regressors": "regressor",
            "Clusterers": "clusterer",
            "Transformers": "transformer",
            "Outlier Detectors": "outlier_detector",
        }

        tabs = st.tabs(list(categories.keys()))

        for tab, (cat_name, cat_type) in zip(tabs, categories.items(), strict=True):
            with tab:
                entries = explorer.list_by_type(cat_type)
                if entries:
                    # Group by module
                    by_module = {}
                    for entry in entries:
                        module_short = (
                            entry.module.split(".")[-1] if "." in entry.module else entry.module
                        )
                        if module_short not in by_module:
                            by_module[module_short] = []
                        by_module[module_short].append(entry)

                    for module_name in sorted(by_module.keys()):
                        st.markdown(f"**{module_name}**")
                        cols = st.columns(4)
                        for i, entry in enumerate(
                            sorted(by_module[module_name], key=lambda x: x.name)
                        ):
                            with cols[i % 4]:
                                if st.button(entry.name, key=f"btn_{cat_type}_{entry.name}"):
                                    st.session_state.selected_entry = entry.name
                else:
                    st.info(f"No {cat_name.lower()} found")

    # Display selected entry details
    if "selected_entry" in st.session_state and st.session_state.selected_entry:
        st.markdown("---")
        entry = get_api_entry(st.session_state.selected_entry)
        if entry:
            st.subheader(f"📋 {entry.name}")
            display_entry_details(entry)


def display_entry(entry):
    """Display a brief entry summary."""
    st.markdown(f"**Module:** `{entry.module}`")
    st.markdown(f"**Kind:** {entry.kind}")

    if entry.estimator_type:
        st.markdown(f"**Estimator Type:** {entry.estimator_type}")

    st.markdown("**Short Description:**")
    st.markdown(entry.short_doc if entry.short_doc else "*No description available*")

    if entry.signature:
        st.markdown("**Signature:**")
        st.code(f"{entry.name}{entry.signature}", language="python")

    if entry.tags:
        st.markdown("**Tags:** " + ", ".join(f"`{tag}`" for tag in entry.tags))


def display_entry_details(entry):
    """Display detailed entry information."""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"**Module:** `{entry.module}`")
        st.markdown(f"**Type:** {entry.kind}")

        if entry.estimator_type:
            st.markdown(f"**Estimator Type:** {entry.estimator_type}")

        st.markdown("### Description")
        st.markdown(entry.short_doc if entry.short_doc else "*No description available*")

    with col2:
        st.markdown("### Import")
        st.code(f"from {entry.module} import {entry.name}", language="python")

        if entry.tags:
            st.markdown("### Tags")
            st.markdown(" ".join(f"`{tag}`" for tag in entry.tags))

    # Signature
    if entry.signature:
        st.markdown("### Signature")
        st.code(f"{entry.name}{entry.signature}", language="python")

    # Parameters
    if entry.parameters:
        st.markdown("### Parameters")

        param_data = []
        for param in entry.parameters:
            param_data.append(
                {
                    "Name": param["name"],
                    "Default": param["default"] if param["default"] else "-",
                    "Description": param.get("description", "")[:100],
                }
            )

        st.dataframe(pd.DataFrame(param_data), hide_index=True, width="stretch")

    # Example
    st.markdown("### Quick Example")
    example = generate_example_snippet(entry)
    st.code(example, language="python")

    # Link to docs
    doc_url = f"https://scikit-learn.org/stable/modules/generated/{entry.module}.{entry.name}.html"
    st.markdown(f"[📚 Full Documentation]({doc_url})")


# Quick reference section
def quick_reference():
    """Show quick reference for common operations."""
    st.markdown("---")
    st.subheader("Quick Reference")

    references = {
        "Training": """
```python
model.fit(X_train, y_train)
```
        """,
        "Prediction": """
```python
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)  # Classification
```
        """,
        "Evaluation": """
```python
score = model.score(X_test, y_test)
```
        """,
        "Parameters": """
```python
params = model.get_params()
model.set_params(param=value)
```
        """,
        "Pipeline": """
```python
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])
```
        """,
        "Cross-Validation": """
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
```
        """,
    }

    cols = st.columns(3)
    for i, (name, code) in enumerate(references.items()):
        with cols[i % 3]:
            st.markdown(f"**{name}**")
            st.markdown(code)


if __name__ == "__main__":
    main()
