"""Model Inspection.

This page covers:
- Feature importance
- Permutation importance
- Partial Dependence Plots (PDP)
- Individual Conditional Expectation (ICE)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

st.set_page_config(page_title="Model Inspection", page_icon="🔬", layout="wide")

from skplay.core.datasets import get_dataset
from skplay.core.evaluation import plot_feature_importance, plot_permutation_importance
from skplay.ui.level import get_level, level_selector


def main():
    st.title("🔬 Model Inspection")

    with st.sidebar:
        level = level_selector()

    st.markdown("""
    Understand what your model learned and why it makes certain predictions.

    [📚 sklearn User Guide: Inspection](https://scikit-learn.org/stable/inspection.html)
    """)

    topic = st.radio(
        "Topic",
        options=["feature_importance", "permutation_importance", "partial_dependence"],
        format_func=lambda x: x.replace("_", " ").title(),
        horizontal=True,
    )

    st.markdown("---")

    if topic == "feature_importance":
        feature_importance_section()
    elif topic == "permutation_importance":
        permutation_importance_section()
    else:
        partial_dependence_section()


def feature_importance_section():
    """Feature importance from tree-based models."""
    st.header("Feature Importance")

    st.markdown("""
    Tree-based models (Random Forest, Gradient Boosting) provide built-in feature importance
    based on how much each feature reduces impurity across all trees.

    **Note:** This is model-specific and can be biased toward high-cardinality features.
    """)

    # Interactive demo
    st.subheader("Interactive Demo")

    col1, col2 = st.columns([1, 2])

    with col1:
        task_type = st.radio("Task", ["classification", "regression"], key="fi_task")

        if task_type == "classification":
            dataset_name = st.selectbox("Dataset", ["iris", "breast_cancer", "wine"], key="fi_data")
        else:
            dataset_name = st.selectbox("Dataset", ["diabetes", "california_housing"], key="fi_data_reg")

        n_estimators = st.slider("Number of Trees", 10, 200, 100, key="fi_trees")

    with col2:
        if st.button("Compute Feature Importance", key="fi_run"):
            with st.spinner("Training model..."):
                data = get_dataset(dataset_name)
                X, y = data.X, data.y

                if task_type == "classification":
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", model),
                ])

                pipeline.fit(X, y)

                # Get feature importance
                importances = pipeline.named_steps["model"].feature_importances_
                feature_names = X.columns.tolist()

                # Sort and display
                sorted_idx = np.argsort(importances)[::-1]

                fig, ax = plt.subplots(figsize=(10, max(5, len(feature_names) * 0.3)))
                ax.barh(range(len(sorted_idx)), importances[sorted_idx][::-1])
                ax.set_yticks(range(len(sorted_idx)))
                ax.set_yticklabels([feature_names[i] for i in sorted_idx[::-1]])
                ax.set_xlabel("Importance")
                ax.set_title("Feature Importance (Mean Decrease in Impurity)")
                st.pyplot(fig)

                # Table
                importance_df = pd.DataFrame({
                    "Feature": [feature_names[i] for i in sorted_idx],
                    "Importance": importances[sorted_idx],
                })
                st.dataframe(importance_df, hide_index=True)

    with st.expander("How It Works"):
        st.markdown("""
        For tree-based models, feature importance is calculated as:

        1. For each tree in the forest
        2. For each node using feature i
        3. Compute the weighted reduction in impurity (Gini or entropy for classification, variance for regression)
        4. Sum across all nodes and trees
        5. Normalize to sum to 1

        **Limitations:**
        - Biased toward continuous and high-cardinality features
        - Doesn't show direction of effect
        - Correlated features can share importance
        """)


def permutation_importance_section():
    """Permutation importance."""
    st.header("Permutation Importance")

    st.markdown("""
    Measures feature importance by randomly shuffling each feature and measuring
    the drop in model performance. Works for any model.

    **Advantages:**
    - Model-agnostic
    - Computed on validation data
    - Shows actual impact on predictions
    """)

    st.subheader("Interactive Demo")

    col1, col2 = st.columns([1, 2])

    with col1:
        dataset_name = st.selectbox("Dataset", ["iris", "breast_cancer"], key="pi_data")
        n_repeats = st.slider("Permutation Repeats", 5, 30, 10, key="pi_repeats")

    with col2:
        if st.button("Compute Permutation Importance", key="pi_run"):
            with st.spinner("Computing (this may take a moment)..."):
                from sklearn.inspection import permutation_importance as perm_imp

                data = get_dataset(dataset_name)
                X, y = data.X, data.y

                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )

                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
                ])

                model.fit(X_train, y_train)

                # Compute permutation importance on test set
                result = perm_imp(model, X_test, y_test, n_repeats=n_repeats, random_state=42)

                # Visualization
                sorted_idx = result.importances_mean.argsort()[::-1]
                feature_names = X.columns.tolist()

                fig, ax = plt.subplots(figsize=(10, max(5, len(feature_names) * 0.3)))
                ax.boxplot(
                    result.importances[sorted_idx].T,
                    vert=False,
                    labels=[feature_names[i] for i in sorted_idx]
                )
                ax.set_xlabel("Decrease in Accuracy")
                ax.set_title("Permutation Importance (on test set)")
                st.pyplot(fig)

                # Table
                importance_df = pd.DataFrame({
                    "Feature": [feature_names[i] for i in sorted_idx],
                    "Mean Importance": result.importances_mean[sorted_idx],
                    "Std": result.importances_std[sorted_idx],
                })
                st.dataframe(importance_df, hide_index=True)

    with st.expander("How It Works"):
        st.markdown("""
        For each feature:

        1. Record baseline score on test data
        2. Randomly shuffle the feature values (break relationship with target)
        3. Compute new score
        4. Importance = baseline score - shuffled score
        5. Repeat multiple times for confidence intervals

        **Benefits over tree-based importance:**
        - Works with any model
        - Computed on held-out data (shows generalization impact)
        - Less biased toward high-cardinality features

        **Note:** Correlated features can still share importance.
        """)


def partial_dependence_section():
    """Partial Dependence Plots."""
    st.header("Partial Dependence Plots")

    level = get_level()

    st.markdown("""
    Shows the marginal effect of a feature on predictions, averaging over all other features.

    Useful for understanding the direction and shape of a feature's influence.
    """)

    st.subheader("Interactive Demo")

    col1, col2 = st.columns([1, 2])

    with col1:
        dataset_name = st.selectbox("Dataset", ["diabetes", "california_housing"], key="pdp_data")

        data = get_dataset(dataset_name)
        feature_names = data.X.columns.tolist()

        selected_feature = st.selectbox("Feature", feature_names, key="pdp_feature")

        if level == "advanced":
            show_ice = st.checkbox("Show ICE (Individual Conditional Expectation)", key="pdp_ice")
        else:
            show_ice = False

    with col2:
        if st.button("Generate PDP", key="pdp_run"):
            with st.spinner("Computing..."):
                from sklearn.inspection import PartialDependenceDisplay

                X, y = data.X, data.y

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                feature_idx = feature_names.index(selected_feature)

                fig, ax = plt.subplots(figsize=(10, 6))

                kind = "both" if show_ice else "average"

                PartialDependenceDisplay.from_estimator(
                    model, X_train, [feature_idx],
                    kind=kind,
                    ax=ax,
                    random_state=42,
                    n_jobs=-1,
                )

                ax.set_title(f"Partial Dependence: {selected_feature}")
                st.pyplot(fig)

    with st.expander("How It Works"):
        st.markdown("""
        Partial Dependence measures how the model's predictions change as we vary one feature
        while averaging over the distribution of all other features.

        **Calculation:**
        1. For each value x_s of feature S:
        2. Replace feature S with x_s for all samples
        3. Predict for all samples
        4. Average predictions

        **ICE (Individual Conditional Expectation):**
        - Shows individual curves for each sample
        - Reveals heterogeneity in feature effects
        - Useful for detecting interactions

        **Interpretation:**
        - Slope shows sensitivity of prediction to feature
        - Shape shows linear vs nonlinear relationship
        - Spread in ICE curves indicates interactions
        """)

    if level in ("intermediate", "advanced"):
        with st.expander("2D Partial Dependence"):
            st.markdown("""
            For two features, we can create a 2D PDP showing their joint effect:

            ```python
            from sklearn.inspection import PartialDependenceDisplay

            PartialDependenceDisplay.from_estimator(
                model, X_train,
                [(feature1, feature2)],  # Pair of features
                kind="average",
            )
            ```

            This reveals interactions between features when the effect of one
            feature depends on the value of another.
            """)


if __name__ == "__main__":
    main()
