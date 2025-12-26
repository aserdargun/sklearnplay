"""Supervised Learning - Classification and Regression.

This page covers:
- Linear Models
- Support Vector Machines
- Nearest Neighbors
- Decision Trees
- Ensemble Methods
- Neural Networks
- And more...
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Supervised Learning", page_icon="🎯", layout="wide")

from skplay.core.datasets import DatasetRegistry, get_dataset
from skplay.core.estimators import create_estimator
from skplay.core.evaluation import (
    evaluate_model,
    plot_actual_vs_predicted,
    plot_confusion_matrix,
    plot_learning_curve,
    plot_residuals,
    plot_roc_curve,
    run_cross_validation,
)
from skplay.core.preprocessing import PreprocessingBuilder, encode_target, identify_column_types
from skplay.core.snippets import generate_code_snippet
from skplay.ui.components import (
    create_download_button,
    show_code_snippet,
    show_data_preview,
    show_dataset_card,
    show_estimator_selector,
    show_metrics_table,
    show_parameter_controls,
    show_preprocessing_controls,
    show_split_controls,
    show_training_button,
)
from skplay.ui.level import get_level, get_level_config, level_selector


def main():
    st.title("🎯 Supervised Learning")

    st.markdown("""
    Supervised learning uses labeled data to train models that predict outcomes.

    **Classification** predicts discrete categories (spam/not spam, species, etc.)

    **Regression** predicts continuous values (price, temperature, etc.)

    [📚 sklearn User Guide: Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html)
    """)

    # Level selector in sidebar
    with st.sidebar:
        level_selector()
        st.markdown("---")

    # Main workflow tabs
    tabs = st.tabs(["📊 Data", "🔧 Preprocessing", "🤖 Model", "📈 Results", "💡 Learn More"])

    with tabs[0]:
        data_result = data_section()

    with tabs[1]:
        if data_result:
            preproc_config = preprocessing_section(data_result)
        else:
            st.info("👆 Select a dataset first")
            preproc_config = None

    with tabs[2]:
        if data_result and preproc_config is not None:
            model_result = model_section(data_result, preproc_config)
        else:
            st.info("👆 Configure preprocessing first")
            model_result = None

    with tabs[3]:
        if model_result:
            results_section(model_result, data_result)
        else:
            st.info("👆 Train a model first")

    with tabs[4]:
        learn_more_section()


@st.cache_data
def load_dataset(name: str):
    """Load and cache a dataset."""
    return get_dataset(name)


def data_section():
    """Dataset selection and preview section."""
    st.header("Select Dataset")

    # Task type selection
    task_type = st.radio(
        "Task Type",
        options=["classification", "regression"],
        format_func=lambda x: x.title(),
        horizontal=True,
        key="task_type",
    )

    # Dataset source
    data_source = st.radio(
        "Data Source",
        options=["toy_dataset", "upload"],
        format_func=lambda x: "Toy Dataset" if x == "toy_dataset" else "Upload CSV",
        horizontal=True,
        key="data_source",
    )

    if data_source == "toy_dataset":
        # Filter datasets by task type
        available = DatasetRegistry.list_by_task(task_type)

        if not available:
            st.warning(f"No datasets available for {task_type}")
            return None

        # Group by domain
        domains = {}
        for name in available:
            card = DatasetRegistry.get_card(name)
            if card.domain not in domains:
                domains[card.domain] = []
            domains[card.domain].append(name)

        # Select domain first
        selected_domain = st.selectbox(
            "Domain",
            options=list(domains.keys()),
            format_func=lambda x: x.title(),
            key="selected_domain",
        )

        # Then dataset
        selected_dataset = st.selectbox(
            "Dataset",
            options=domains[selected_domain],
            key="selected_dataset",
        )

        if selected_dataset:
            result = load_dataset(selected_dataset)
            show_dataset_card(result.card)
            show_data_preview(result.X, result.y)

            st.session_state.current_data = result
            st.session_state.current_task = task_type
            return result

    else:
        # CSV Upload
        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            key="csv_upload",
        )

        if uploaded_file:
            from skplay.core.upload import create_dataset_from_upload

            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(), use_container_width=True)

            # Target column selection
            target_col = st.selectbox(
                "Target Column",
                options=[None] + list(df.columns),
                format_func=lambda x: "(No target)" if x is None else x,
                key="target_col",
            )

            if st.button("Create Dataset"):
                result = create_dataset_from_upload(
                    df,
                    target_column=target_col,
                    task_type=task_type,
                    dataset_name=uploaded_file.name.replace(".csv", ""),
                )
                st.session_state.current_data = result
                st.session_state.current_task = task_type
                st.success("Dataset created!")
                return result

    return st.session_state.get("current_data")


def preprocessing_section(data_result):
    """Preprocessing configuration section."""
    st.header("Configure Preprocessing")

    X = data_result.X
    task_type = data_result.card.task_type

    # Identify column types
    numeric_cols, categorical_cols = identify_column_types(X)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Numeric columns:** {len(numeric_cols)}")
        if numeric_cols:
            st.caption(", ".join(numeric_cols[:5]) + ("..." if len(numeric_cols) > 5 else ""))
    with col2:
        st.markdown(f"**Categorical columns:** {len(categorical_cols)}")
        if categorical_cols:
            st.caption(
                ", ".join(categorical_cols[:5]) + ("..." if len(categorical_cols) > 5 else "")
            )

    # Show preprocessing controls
    config = show_preprocessing_controls(
        numeric_cols, categorical_cols, task_type, key_prefix="main_preproc"
    )

    # Store in session state
    config["numeric_cols"] = numeric_cols
    config["categorical_cols"] = categorical_cols
    st.session_state.preproc_config = config

    return config


def model_section(data_result, preproc_config):
    """Model selection and training section."""
    st.header("Select and Train Model")

    level = get_level()
    task_type = data_result.card.task_type

    # Split controls
    split_config = show_split_controls(len(data_result.X), key_prefix="main_split")

    st.markdown("---")

    # Estimator selection
    estimator_info = show_estimator_selector(task_type, level, key="main_estimator")

    if not estimator_info:
        return None

    st.markdown("---")

    # Parameter controls
    params = show_parameter_controls(estimator_info, prefix="", key_prefix="main_params")

    st.markdown("---")

    # Training button
    if show_training_button(key="main_train"):
        with st.spinner("Training model..."):
            result = train_model(data_result, preproc_config, estimator_info, params, split_config)
            st.session_state.training_result = result
            st.success("Training complete!")
            return result

    return st.session_state.get("training_result")


def train_model(data_result, preproc_config, estimator_info, params, split_config):
    """Train the model with given configuration."""
    X = data_result.X
    y = data_result.y
    task_type = data_result.card.task_type

    # Encode target if needed
    y_encoded, label_encoder = encode_target(y, task_type)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=split_config["test_size"],
        random_state=split_config["random_state"],
    )

    # Build preprocessing pipeline
    preproc_builder = PreprocessingBuilder(
        numeric_imputer=preproc_config.get("numeric_imputer", "median"),
        numeric_scaler=preproc_config.get("numeric_scaler", "standard"),
        categorical_imputer=preproc_config.get("categorical_imputer", "most_frequent"),
        categorical_encoder=preproc_config.get("categorical_encoder", "onehot"),
        feature_selector=preproc_config.get("feature_selector", "none"),
        feature_selector_k=preproc_config.get("feature_selector_k", 10),
        dim_reducer=preproc_config.get("dim_reducer", "none"),
        dim_reducer_n_components=preproc_config.get("dim_reducer_n_components", 0.95),
        task_type=task_type,
    )

    preprocessing_pipeline, numeric_cols, categorical_cols = preproc_builder.build_full_pipeline(
        X_train,
        force_categorical=preproc_config.get("categorical_cols"),
        force_numeric=preproc_config.get("numeric_cols"),
    )

    # Create estimator
    estimator = create_estimator(task_type, estimator_info.name, **params)

    # Build full pipeline
    full_pipeline = Pipeline(
        [
            ("preprocessing", preprocessing_pipeline),
            ("estimator", estimator),
        ]
    )

    # Fit
    full_pipeline.fit(X_train, y_train)

    # Evaluate
    eval_result = evaluate_model(full_pipeline, X_train, y_train, X_test, y_test, task_type)

    # Cross-validation if requested
    cv_result = None
    if split_config.get("use_cv"):
        cv_result = run_cross_validation(
            full_pipeline,
            X,
            y_encoded,
            cv=split_config.get("cv_folds", 5),
            task_type=task_type,
        )

    return {
        "pipeline": full_pipeline,
        "eval_result": eval_result,
        "cv_result": cv_result,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "label_encoder": label_encoder,
        "estimator_info": estimator_info,
        "params": params,
        "preproc_config": preproc_config,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


def results_section(model_result, data_result):
    """Display training results and visualizations."""
    st.header("Results")

    eval_result = model_result["eval_result"]
    task_type = eval_result["task_type"]
    config = get_level_config()

    # Metrics
    col1, col2 = st.columns(2)

    with col1:
        show_metrics_table(eval_result["train_metrics"], "Training Metrics")

    with col2:
        show_metrics_table(eval_result["test_metrics"], "Test Metrics")

    # Cross-validation results
    if model_result["cv_result"] and "error" not in model_result["cv_result"]:
        st.markdown("#### Cross-Validation Results")
        cv = model_result["cv_result"]
        cv_metrics = {
            k.replace("test_", ""): f"{v['mean']:.4f} (+/- {v['std']:.4f})"
            for k, v in cv.items()
            if k.startswith("test_") and isinstance(v, dict)
        }
        st.json(cv_metrics)

    st.markdown("---")

    # Visualizations
    st.subheader("Visualizations")

    if task_type == "classification":
        viz_tabs = st.tabs(["Confusion Matrix", "ROC Curve", "Learning Curve"])

        with viz_tabs[0]:
            if "confusion_matrix" in eval_result:
                # Get class labels
                labels = None
                if model_result["label_encoder"]:
                    labels = model_result["label_encoder"].classes_.tolist()
                elif data_result.y is not None:
                    labels = data_result.y.unique().tolist()

                fig = plot_confusion_matrix(eval_result["confusion_matrix"], labels)
                st.pyplot(fig)

        with viz_tabs[1]:
            if eval_result.get("y_proba") is not None:
                try:
                    fig = plot_roc_curve(model_result["y_test"], eval_result["y_proba"])
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not plot ROC curve: {e}")
            else:
                st.info("ROC curve requires probability predictions")

        with viz_tabs[2]:
            if config["show_learning_curve"]:
                with st.spinner("Computing learning curve..."):
                    try:
                        fig = plot_learning_curve(
                            model_result["pipeline"],
                            pd.concat([model_result["X_train"], model_result["X_test"]]),
                            np.concatenate([model_result["y_train"], model_result["y_test"]]),
                        )
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"Could not plot learning curve: {e}")

    elif task_type == "regression":
        viz_tabs = st.tabs(["Residuals", "Actual vs Predicted", "Learning Curve"])

        with viz_tabs[0]:
            fig = plot_residuals(model_result["y_test"], eval_result["y_pred"])
            st.pyplot(fig)

        with viz_tabs[1]:
            fig = plot_actual_vs_predicted(model_result["y_test"], eval_result["y_pred"])
            st.pyplot(fig)

        with viz_tabs[2]:
            if config["show_learning_curve"]:
                with st.spinner("Computing learning curve..."):
                    try:
                        fig = plot_learning_curve(
                            model_result["pipeline"],
                            pd.concat([model_result["X_train"], model_result["X_test"]]),
                            np.concatenate([model_result["y_train"], model_result["y_test"]]),
                            scoring="r2",
                        )
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"Could not plot learning curve: {e}")

    st.markdown("---")

    # Code snippet
    if config["show_code_snippet"]:
        st.subheader("Reproducible Code")

        code = generate_code_snippet(
            model_result["pipeline"],
            X_train_shape=model_result["X_train"].shape,
            y_train_shape=model_result["y_train"].shape,
            dataset_name=data_result.card.name,
            task_type=task_type,
            numeric_cols=model_result["numeric_cols"],
            categorical_cols=model_result["categorical_cols"],
        )

        show_code_snippet(code)

    # Model export
    if config["show_model_export"]:
        st.markdown("---")
        create_download_button(
            model_result["pipeline"],
            filename=f"{data_result.card.name}_model.joblib",
        )


def learn_more_section():
    """Educational content about supervised learning."""
    st.header("Learn More")

    level = get_level()

    st.markdown("""
    ### Key Concepts

    **Supervised learning** trains models on labeled data to make predictions.
    The model learns the relationship between input features (X) and target values (y).
    """)

    with st.expander("Classification vs Regression"):
        st.markdown("""
        **Classification** predicts discrete categories:
        - Binary: yes/no, spam/not spam
        - Multiclass: species, category, sentiment

        **Regression** predicts continuous values:
        - Prices, temperatures, quantities

        The choice depends on your target variable's nature.
        """)

    with st.expander("Train/Test Split"):
        st.markdown("""
        We split data to evaluate model performance on unseen data:

        - **Training set**: Used to fit the model
        - **Test set**: Used to evaluate generalization

        Common splits: 80/20 or 70/30 (train/test)

        **Why?** A model that memorizes training data (overfitting)
        will perform poorly on new data.
        """)

    with st.expander("Common Algorithms"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Classification:**
            - Logistic Regression
            - Decision Trees
            - Random Forest
            - SVM
            - K-Nearest Neighbors
            """)

        with col2:
            st.markdown("""
            **Regression:**
            - Linear Regression
            - Ridge / Lasso
            - Decision Tree Regressor
            - Random Forest Regressor
            - Gradient Boosting
            """)

    if level in ("intermediate", "advanced"):
        with st.expander("Cross-Validation"):
            st.markdown("""
            Cross-validation provides more robust performance estimates:

            1. Split data into K folds
            2. Train on K-1 folds, test on remaining fold
            3. Repeat K times with different test folds
            4. Average the results

            This reduces variance in performance estimates and uses all data for both training and testing.
            """)

    if level == "advanced":
        with st.expander("Bias-Variance Tradeoff"):
            st.markdown("""
            - **Bias**: Error from oversimplified assumptions
            - **Variance**: Error from sensitivity to training data

            Complex models: Low bias, high variance (overfitting risk)
            Simple models: High bias, low variance (underfitting risk)

            Goal: Find the sweet spot that minimizes total error.
            """)


if __name__ == "__main__":
    main()
