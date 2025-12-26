"""Model Selection and Evaluation.

This page covers:
- Cross-validation
- Hyperparameter tuning (Grid Search, Random Search, Successive Halving)
- Metrics and scoring
- Validation curves
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, validation_curve

st.set_page_config(page_title="Model Selection", page_icon="🎚️", layout="wide")

from skplay.core.datasets import DatasetRegistry, get_dataset
from skplay.core.preprocessing import PreprocessingBuilder, identify_column_types, encode_target
from skplay.core.estimators import get_estimators_for_task, create_estimator
from skplay.core.tuning import get_param_grid, run_grid_search, run_random_search, format_search_results, HALVING_AVAILABLE
from skplay.core.evaluation import plot_learning_curve, plot_validation_curve
from skplay.ui.level import get_level, get_level_config, level_selector
from skplay.ui.components import show_dataset_card, show_estimator_selector


def main():
    st.title("🎚️ Model Selection and Evaluation")

    st.markdown("""
    Learn how to properly evaluate models and tune hyperparameters.

    [📚 sklearn User Guide: Model Selection](https://scikit-learn.org/stable/model_selection.html)
    """)

    with st.sidebar:
        level = level_selector()

    # Topic selection
    topic = st.radio(
        "Topic",
        options=["cross_validation", "hyperparameter_tuning", "learning_curves", "metrics"],
        format_func=lambda x: x.replace("_", " ").title(),
        horizontal=True,
    )

    st.markdown("---")

    if topic == "cross_validation":
        cross_validation_section()
    elif topic == "hyperparameter_tuning":
        hyperparameter_tuning_section()
    elif topic == "learning_curves":
        learning_curves_section()
    else:
        metrics_section()


def cross_validation_section():
    """Cross-validation demonstration."""
    st.header("Cross-Validation")

    st.markdown("""
    Cross-validation provides more reliable performance estimates by testing on multiple data splits.

    **K-Fold CV:** Split data into K folds, use each as test set once while training on the rest.
    """)

    # Interactive demo
    st.subheader("Interactive Demo")

    col1, col2 = st.columns([1, 2])

    with col1:
        dataset_name = st.selectbox(
            "Dataset",
            options=["iris", "wine", "breast_cancer"],
            key="cv_dataset",
        )

        n_folds = st.slider("Number of Folds (K)", 2, 10, 5, key="cv_folds")

        estimator_name = st.selectbox(
            "Estimator",
            options=["Logistic Regression", "Random Forest", "SVM (RBF Kernel)"],
            key="cv_estimator",
        )

    with col2:
        if st.button("Run Cross-Validation", key="cv_run"):
            with st.spinner("Running CV..."):
                # Load data
                data = get_dataset(dataset_name)
                X, y = data.X, data.y

                # Encode target
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)

                # Create estimator
                estimator = create_estimator("classification", estimator_name)

                # Run CV
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import Pipeline

                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("estimator", estimator),
                ])

                scores = cross_val_score(pipeline, X, y_encoded, cv=n_folds, scoring="accuracy")

                # Display results
                st.markdown("#### Results")

                st.metric("Mean Accuracy", f"{scores.mean():.4f}")
                st.metric("Std Deviation", f"{scores.std():.4f}")

                # Show fold-by-fold
                fold_df = pd.DataFrame({
                    "Fold": range(1, n_folds + 1),
                    "Accuracy": scores,
                })
                st.dataframe(fold_df, hide_index=True)

                # Visualization
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(range(1, n_folds + 1), scores)
                ax.axhline(y=scores.mean(), color='r', linestyle='--', label=f'Mean: {scores.mean():.3f}')
                ax.set_xlabel("Fold")
                ax.set_ylabel("Accuracy")
                ax.set_title("Cross-Validation Scores")
                ax.legend()
                st.pyplot(fig)

    # Explanation
    level = get_level()

    with st.expander("How K-Fold CV Works"):
        st.markdown("""
        1. Split data into K equal folds
        2. For each fold i = 1 to K:
           - Use fold i as test set
           - Use remaining K-1 folds as training set
           - Train model and evaluate on test fold
        3. Average all K scores for final estimate

        **Benefits:**
        - Uses all data for both training and testing
        - Reduces variance in performance estimates
        - More reliable than single train/test split
        """)

    if level in ("intermediate", "advanced"):
        with st.expander("Stratified K-Fold"):
            st.markdown("""
            For classification, stratified K-Fold preserves class proportions in each fold.

            This is important when classes are imbalanced to ensure each fold has representative samples of all classes.

            ```python
            from sklearn.model_selection import StratifiedKFold

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=skf)
            ```
            """)


def hyperparameter_tuning_section():
    """Hyperparameter tuning demonstration."""
    st.header("Hyperparameter Tuning")

    level = get_level()
    config = get_level_config()

    st.markdown("""
    Find optimal hyperparameters through systematic search.
    """)

    # Methods overview
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        #### Grid Search
        - Exhaustive search over parameter grid
        - Guaranteed to find best in grid
        - Can be slow with many parameters
        """)

    with col2:
        st.markdown("""
        #### Random Search
        - Random sampling from distributions
        - Often finds good solutions faster
        - Better for high-dimensional spaces
        """)

    if config["show_halving_search"]:
        with col3:
            st.markdown("""
            #### Successive Halving
            - Progressive resource allocation
            - Eliminates poor candidates early
            - Very efficient for large grids
            """)

    st.markdown("---")

    # Interactive demo
    st.subheader("Interactive Demo")

    col1, col2 = st.columns([1, 2])

    with col1:
        dataset_name = st.selectbox(
            "Dataset",
            options=["iris", "breast_cancer"],
            key="tune_dataset",
        )

        search_method = st.selectbox(
            "Search Method",
            options=["grid_search", "random_search"] + (["halving_search"] if HALVING_AVAILABLE and config["show_halving_search"] else []),
            format_func=lambda x: x.replace("_", " ").title(),
            key="tune_method",
        )

        cv_folds = st.slider("CV Folds", 2, 5, 3, key="tune_cv")

    with col2:
        st.markdown("**Parameter Grid (Random Forest):**")
        st.code("""
{
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}
        """)

        if st.button("Run Search", key="tune_run"):
            with st.spinner("Searching (this may take a minute)..."):
                # Load data
                data = get_dataset(dataset_name)
                X, y = data.X, data.y

                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=0.2, random_state=42
                )

                # Create pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import Pipeline
                from sklearn.ensemble import RandomForestClassifier

                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", RandomForestClassifier(random_state=42)),
                ])

                # Parameter grid
                param_grid = {
                    "clf__n_estimators": [50, 100, 200],
                    "clf__max_depth": [None, 5, 10],
                    "clf__min_samples_split": [2, 5],
                }

                # Run search
                if search_method == "grid_search":
                    from sklearn.model_selection import GridSearchCV
                    search = GridSearchCV(
                        pipeline, param_grid, cv=cv_folds, scoring="accuracy", n_jobs=-1
                    )
                else:
                    from sklearn.model_selection import RandomizedSearchCV
                    search = RandomizedSearchCV(
                        pipeline, param_grid, n_iter=12, cv=cv_folds,
                        scoring="accuracy", n_jobs=-1, random_state=42
                    )

                search.fit(X_train, y_train)

                # Results
                st.markdown("#### Results")

                st.metric("Best Score (CV)", f"{search.best_score_:.4f}")
                st.metric("Test Score", f"{search.score(X_test, y_test):.4f}")

                st.markdown("**Best Parameters:**")
                st.json({k.replace("clf__", ""): v for k, v in search.best_params_.items()})

                # Top results
                results = format_search_results(search, top_n=5)
                results_df = pd.DataFrame([
                    {
                        "Rank": r["rank"],
                        "Mean Score": f"{r['mean_test_score']:.4f}",
                        "Std": f"{r['std_test_score']:.4f}",
                        **{k.replace("clf__", ""): v for k, v in r["params"].items()}
                    }
                    for r in results
                ])
                st.dataframe(results_df, hide_index=True)

    with st.expander("Code Example"):
        st.code("""
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
}

# Create and run grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
        """, language="python")


def learning_curves_section():
    """Learning and validation curves."""
    st.header("Learning and Validation Curves")

    st.markdown("""
    Visualize model behavior as training data or hyperparameters change.
    """)

    curve_type = st.radio(
        "Curve Type",
        options=["learning", "validation"],
        format_func=lambda x: f"{x.title()} Curve",
        horizontal=True,
    )

    st.markdown("---")

    if curve_type == "learning":
        st.subheader("Learning Curve")

        st.markdown("""
        Shows how training and validation scores change with training set size.

        **Interpretation:**
        - High training, low validation = overfitting (need more data or regularization)
        - Both low = underfitting (need more complex model)
        - Both converging high = good fit
        """)

        col1, col2 = st.columns([1, 2])

        with col1:
            dataset = st.selectbox("Dataset", ["iris", "breast_cancer"], key="lc_data")
            estimator = st.selectbox("Model", ["Random Forest", "Logistic Regression", "SVM (RBF Kernel)"], key="lc_est")

        with col2:
            if st.button("Generate Learning Curve", key="lc_run"):
                with st.spinner("Computing..."):
                    data = get_dataset(dataset)
                    X, y = data.X, data.y

                    from sklearn.preprocessing import LabelEncoder, StandardScaler
                    from sklearn.pipeline import Pipeline

                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y)

                    est = create_estimator("classification", estimator)
                    pipeline = Pipeline([
                        ("scaler", StandardScaler()),
                        ("estimator", est),
                    ])

                    fig = plot_learning_curve(pipeline, X, y_encoded, cv=5)
                    st.pyplot(fig)

    else:
        st.subheader("Validation Curve")

        st.markdown("""
        Shows how training and validation scores change with a hyperparameter.

        **Interpretation:**
        - Training increases but validation decreases = overfitting
        - Both increase together = good region
        - Peak validation score = optimal parameter value
        """)

        col1, col2 = st.columns([1, 2])

        with col1:
            dataset = st.selectbox("Dataset", ["iris", "breast_cancer"], key="vc_data")
            param_name = st.selectbox("Parameter", ["n_estimators", "max_depth"], key="vc_param")

        with col2:
            if st.button("Generate Validation Curve", key="vc_run"):
                with st.spinner("Computing..."):
                    data = get_dataset(dataset)
                    X, y = data.X, data.y

                    from sklearn.preprocessing import LabelEncoder, StandardScaler
                    from sklearn.pipeline import Pipeline
                    from sklearn.ensemble import RandomForestClassifier

                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y)

                    pipeline = Pipeline([
                        ("scaler", StandardScaler()),
                        ("clf", RandomForestClassifier(random_state=42)),
                    ])

                    if param_name == "n_estimators":
                        param_range = np.array([10, 25, 50, 100, 200, 500])
                    else:
                        param_range = np.array([1, 3, 5, 10, 15, 20, None])
                        param_range = param_range[:-1].astype(int)  # Remove None for now

                    fig = plot_validation_curve(
                        pipeline, X, y_encoded,
                        param_name=f"clf__{param_name}",
                        param_range=param_range,
                        cv=5,
                    )
                    st.pyplot(fig)


def metrics_section():
    """Metrics and scoring functions."""
    st.header("Metrics and Scoring")

    level = get_level()

    st.markdown("""
    Choose the right metric for your problem. Different metrics emphasize different aspects of model performance.
    """)

    metric_type = st.radio(
        "Task Type",
        options=["classification", "regression"],
        horizontal=True,
    )

    if metric_type == "classification":
        st.markdown("### Classification Metrics")

        metrics_data = [
            {"Metric": "Accuracy", "Formula": "(TP + TN) / Total", "Use When": "Classes are balanced", "Range": "0 to 1"},
            {"Metric": "Precision", "Formula": "TP / (TP + FP)", "Use When": "False positives are costly", "Range": "0 to 1"},
            {"Metric": "Recall", "Formula": "TP / (TP + FN)", "Use When": "False negatives are costly", "Range": "0 to 1"},
            {"Metric": "F1 Score", "Formula": "2 * P * R / (P + R)", "Use When": "Balance P and R", "Range": "0 to 1"},
            {"Metric": "ROC AUC", "Formula": "Area under ROC", "Use When": "Ranking matters", "Range": "0.5 to 1"},
        ]

        st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)

        with st.expander("Confusion Matrix Explained"):
            st.markdown("""
            |  | Predicted Positive | Predicted Negative |
            |--|-------------------|-------------------|
            | **Actual Positive** | TP (True Positive) | FN (False Negative) |
            | **Actual Negative** | FP (False Positive) | TN (True Negative) |

            - **TP**: Correctly identified positives
            - **TN**: Correctly identified negatives
            - **FP**: Type I error (false alarm)
            - **FN**: Type II error (missed detection)
            """)

    else:
        st.markdown("### Regression Metrics")

        metrics_data = [
            {"Metric": "R² Score", "Formula": "1 - SS_res/SS_tot", "Interpretation": "Variance explained (1 = perfect)", "Scale": "≤1"},
            {"Metric": "MAE", "Formula": "mean(|y - ŷ|)", "Interpretation": "Average absolute error", "Scale": "Same as y"},
            {"Metric": "RMSE", "Formula": "√mean((y - ŷ)²)", "Interpretation": "Standard deviation of errors", "Scale": "Same as y"},
            {"Metric": "MAPE", "Formula": "mean(|y - ŷ|/|y|) × 100", "Interpretation": "Percentage error", "Scale": "%"},
        ]

        st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)

        with st.expander("When to Use Each"):
            st.markdown("""
            - **R²**: Overall goodness of fit, comparing to baseline (mean)
            - **MAE**: Robust to outliers, easy to interpret
            - **RMSE**: Penalizes large errors more, in original units
            - **MAPE**: Good for comparing across scales
            """)

    # Scoring in sklearn
    if level in ("intermediate", "advanced"):
        st.markdown("---")
        st.subheader("Using Scoring in sklearn")

        st.code("""
# Available scoring strings
from sklearn.metrics import get_scorer_names
print(get_scorer_names())  # Lists all available scorers

# Using in cross_val_score
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=5)

# Multiple metrics at once
from sklearn.model_selection import cross_validate
cv_results = cross_validate(
    model, X, y,
    scoring=['accuracy', 'precision', 'recall', 'f1'],
    cv=5,
    return_train_score=True
)
        """, language="python")


if __name__ == "__main__":
    main()
