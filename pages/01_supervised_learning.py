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

st.set_page_config(page_title="Supervised Learning", page_icon="images/icon.png", layout="wide")

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


def validate_session_state():
    """Check for and clean up stale session state data."""
    current_data = st.session_state.get("current_data")
    if current_data is not None:
        # Verify that the stored X matches the card metadata
        try:
            card = current_data.card
            X = current_data.X
            if X.shape[0] != card.n_samples or X.shape[1] != card.n_features:
                # Data is corrupted/stale, clear everything
                st.session_state.pop("current_data", None)
                st.session_state.pop("training_result", None)
                st.session_state.pop("preproc_config", None)
                st.warning("Stale data detected and cleared. Please reload your dataset.")
        except Exception:
            # If we can't validate, clear to be safe
            st.session_state.pop("current_data", None)
            st.session_state.pop("training_result", None)
            st.session_state.pop("preproc_config", None)


def main():
    st.title("🎯 Supervised Learning")

    # Clean up any stale session state
    validate_session_state()

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

            # Clear stale results if dataset changed
            old_data = st.session_state.get("current_data")
            if old_data is None or old_data.card.name != result.card.name:
                st.session_state.pop("training_result", None)
                st.session_state.pop("preproc_config", None)

            st.session_state.current_data = result
            st.session_state.current_task = task_type
            return result

    else:
        # CSV Upload
        st.markdown("#### CSV Template")
        st.info(
            "Your CSV should have **feature columns** and optionally a **target column** "
            "for the variable you want to predict."
        )

        # Show template example
        with st.expander("📋 View CSV Template Example"):
            if task_type == "classification":
                template_data = {
                    "feature_1": [1.2, 3.4, 5.6, 7.8],
                    "feature_2": [0.5, 1.5, 2.5, 3.5],
                    "category": ["A", "B", "A", "B"],
                    "target": ["class_1", "class_2", "class_1", "class_2"],
                }
                st.markdown("**Classification example** (target column contains categories):")
            else:
                template_data = {
                    "feature_1": [1.2, 3.4, 5.6, 7.8],
                    "feature_2": [0.5, 1.5, 2.5, 3.5],
                    "category": ["A", "B", "A", "B"],
                    "target": [10.5, 20.3, 15.7, 25.1],
                }
                st.markdown("**Regression example** (target column contains numbers):")

            template_df = pd.DataFrame(template_data)
            st.dataframe(template_df, hide_index=True)

            # Download template button
            csv_template = template_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Template CSV",
                data=csv_template,
                file_name=f"template_{task_type}.csv",
                mime="text/csv",
            )

            st.markdown("""
            **Tips:**
            - First row should be column headers
            - Numeric features: integers or decimals
            - Categorical features: text values
            - Target column: select after upload
            """)

        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            key="csv_upload",
        )

        if uploaded_file:
            # Track which file was last uploaded to detect file changes
            current_file_name = uploaded_file.name
            last_uploaded_file = st.session_state.get("last_uploaded_file")

            if last_uploaded_file != current_file_name:
                # New file uploaded - clear ALL stale data immediately
                st.session_state.pop("current_data", None)
                st.session_state.pop("training_result", None)
                st.session_state.pop("preproc_config", None)
                st.session_state["last_uploaded_file"] = current_file_name
            from skplay.core.upload import (
                create_dataset_from_upload,
                detect_datetime_column,
                get_column_summary,
                validate_upload,
            )

            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(), width="stretch")

            # Show validation warnings
            warnings = validate_upload(df)
            if warnings:
                for warning in warnings:
                    st.warning(warning)

            # Show column summary
            with st.expander("Column Summary", expanded=True):
                summary = get_column_summary(df)
                summary_rows = []
                for col, info in summary.items():
                    row = {
                        "Column": col,
                        "Type": info["inferred_type"],
                        "Dtype": info["dtype"],
                        "Unique": info["n_unique"],
                        "Missing": f"{info['n_missing']} ({info['missing_pct']}%)",
                    }
                    if "mean" in info:
                        row["Stats"] = f"min={info['min']:.2g}, max={info['max']:.2g}, mean={info['mean']:.2g}"
                    else:
                        samples = info.get("sample_values", [])[:3]
                        row["Stats"] = f"samples: {samples}"
                    summary_rows.append(row)
                st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                # Target column selection
                target_col = st.selectbox(
                    "Target Column",
                    options=[None] + list(df.columns),
                    format_func=lambda x: "(No target)" if x is None else x,
                    key="target_col",
                )

            with col2:
                # Datetime column selection (auto-detect)
                detected_datetime = detect_datetime_column(df)
                datetime_options = [None] + list(df.columns)
                default_idx = 0
                if detected_datetime:
                    default_idx = datetime_options.index(detected_datetime)

                datetime_col = st.selectbox(
                    "Datetime Index Column",
                    options=datetime_options,
                    index=default_idx,
                    format_func=lambda x: "(None)" if x is None else x,
                    key="datetime_col",
                    help="Set a timestamp column as the DataFrame index for time series data",
                )

            # Show status of dataset creation
            existing_data = st.session_state.get("current_data")
            if existing_data is None:
                st.warning("Click 'Create Dataset' to prepare your data for training.")
            else:
                # Show info about the currently loaded dataset
                st.info(
                    f"Ready for training: **{existing_data.card.name}** "
                    f"({existing_data.X.shape[0]} samples, {existing_data.X.shape[1]} features)"
                )

            if st.button("Create Dataset"):
                result = create_dataset_from_upload(
                    df,
                    target_column=target_col,
                    datetime_column=datetime_col,
                    task_type=task_type,
                    dataset_name=uploaded_file.name.replace(".csv", ""),
                )
                # DEBUG: Verify data before storing
                st.write(f"[DEBUG] Created dataset - X.shape: {result.X.shape}, columns: {list(result.X.columns)[:5]}")
                st.write(f"[DEBUG] Card: n_samples={result.card.n_samples}, n_features={result.card.n_features}")

                st.session_state.current_data = result
                st.session_state.current_task = task_type
                # Clear stale results from previous dataset
                st.session_state.pop("training_result", None)
                st.session_state.pop("preproc_config", None)
                st.success(
                    f"Dataset created: {result.X.shape[0]} samples, {result.X.shape[1]} features"
                )
                # Don't rerun - let user see debug output and manually proceed
                # st.rerun()

    # DEBUG: Verify data when returning from session state
    stored_data = st.session_state.get("current_data")
    if stored_data is not None:
        st.write(f"[DEBUG data_section] Returning from session - X.shape: {stored_data.X.shape}, columns: {list(stored_data.X.columns)[:5]}")
    return stored_data


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

    # Show current dataset info with actual data verification
    card = data_result.card
    actual_X = data_result.X
    st.info(
        f"**Training Dataset:** {card.name}  \n"
        f"**Card says:** {card.n_samples} samples, {card.n_features} features  \n"
        f"**Actual X:** {actual_X.shape[0]} samples, {actual_X.shape[1]} features  \n"
        f"**Columns:** {list(actual_X.columns)[:5]}{'...' if len(actual_X.columns) > 5 else ''}  \n"
        f"**Task:** {card.task_type} | **Target:** {card.target_name}"
    )

    # Early validation - don't allow training without target
    if data_result.y is None:
        st.error(
            "**No target column selected!** Supervised learning requires a target variable. "
            "Please go back to the **Data** tab, select a target column, and click 'Create Dataset'."
        )
        return None

    # Early validation - don't allow training with 0 features
    if actual_X.shape[1] == 0:
        st.error(
            f"Dataset has 0 feature columns! Card metadata says {card.n_features} features "
            f"but actual data has {actual_X.shape[1]}. Please re-upload your CSV and click 'Create Dataset'."
        )
        return None

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
    X = data_result.X.copy()
    y = data_result.y.copy() if data_result.y is not None else None
    task_type = data_result.card.task_type

    # Supervised learning requires a target column
    if y is None:
        raise ValueError(
            "No target column selected. Supervised learning requires a target variable. "
            "Please go back to the Data tab and select a target column, then click 'Create Dataset'."
        )

    # Validate that actual data matches the card metadata
    card = data_result.card
    if X.shape[0] != card.n_samples or X.shape[1] != card.n_features:
        raise ValueError(
            f"Data mismatch detected! This usually means stale cached data. "
            f"Card says: {card.n_samples} samples, {card.n_features} features. "
            f"Actual X: {X.shape[0]} samples, {X.shape[1]} features. "
            f"Please refresh the page (Ctrl+Shift+R) and re-upload your CSV."
        )

    # Validate input data immediately
    if len(X.columns) == 0:
        raise ValueError(
            f"Dataset has no feature columns! "
            f"Dataset name: {data_result.card.name}, "
            f"X shape: {X.shape}, "
            f"Original columns from card: {[f.name for f in data_result.card.features]}"
        )

    # Ensure column names are strings to avoid type mismatches in sklearn
    X.columns = [str(c) for c in X.columns]

    # Drop rows where target is NaN
    if y is not None:
        if hasattr(y, "isna"):
            valid_mask = ~y.isna()
        else:
            valid_mask = ~pd.isna(y)
        X = X[valid_mask].reset_index(drop=True)
        y = y[valid_mask].reset_index(drop=True)

    # Encode target if needed
    y_encoded, label_encoder = encode_target(y, task_type)

    # DEBUG: Log before split
    st.write(f"[DEBUG train_model] Before split - X.shape: {X.shape}, columns: {list(X.columns)}")
    st.write(f"[DEBUG train_model] X.dtypes: {dict(X.dtypes)}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=split_config["test_size"],
        random_state=split_config["random_state"],
    )

    # DEBUG: Log after split
    st.write(f"[DEBUG train_model] After split - X_train.shape: {X_train.shape}")

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

    # Let the pipeline auto-detect column types from X_train
    # Don't use pre-computed column lists as they may not match after filtering
    preprocessing_pipeline, numeric_cols, categorical_cols = preproc_builder.build_full_pipeline(
        X_train,
    )

    # DEBUG: Log pipeline configuration and test preprocessing output
    st.write(f"[DEBUG train_model] Pipeline built - numeric_cols: {numeric_cols}, categorical_cols: {categorical_cols}")
    try:
        test_transform = preprocessing_pipeline.fit_transform(X_train.head(5))
        st.write(f"[DEBUG train_model] Preprocessing test output shape: {test_transform.shape}")
        if test_transform.shape[1] == 0:
            st.error("[DEBUG train_model] CRITICAL: Preprocessing produces 0 features!")
    except Exception as e:
        st.error(f"[DEBUG train_model] Preprocessing test failed: {e}")

    # Create estimator
    estimator = create_estimator(task_type, estimator_info.name, **params)

    # Build full pipeline
    full_pipeline = Pipeline(
        [
            ("preprocessing", preprocessing_pipeline),
            ("estimator", estimator),
        ]
    )

    # Validate before fitting
    if len(X_train.columns) == 0:
        raise ValueError(
            f"No feature columns found in training data. "
            f"Dataset: {data_result.card.name}, "
            f"Original X shape: {data_result.X.shape}, "
            f"X_train shape: {X_train.shape}, "
            f"numeric_cols: {numeric_cols}, categorical_cols: {categorical_cols}. "
            f"Please ensure your dataset has at least one feature column."
        )

    # Also validate that columns match what ColumnTransformer expects
    all_expected_cols = set(numeric_cols + categorical_cols)
    actual_cols = set(X_train.columns)
    if all_expected_cols and not all_expected_cols.issubset(actual_cols):
        missing = all_expected_cols - actual_cols
        raise ValueError(
            f"Column mismatch! Expected columns {missing} not found in X_train. "
            f"X_train columns: {list(X_train.columns)}, "
            f"numeric_cols: {numeric_cols}, categorical_cols: {categorical_cols}"
        )

    # DEBUG: Final check before fit
    st.write(f"[DEBUG train_model] About to fit - X_train.shape: {X_train.shape}")
    st.write(f"[DEBUG train_model] X_train.columns: {list(X_train.columns)}")

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
            else:
                st.info(
                    "🔒 Learning curves are available at **Intermediate** or **Advanced** level. "
                    "Change your experience level in the sidebar to enable this feature."
                )

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
            else:
                st.info(
                    "🔒 Learning curves are available at **Intermediate** or **Advanced** level. "
                    "Change your experience level in the sidebar to enable this feature."
                )

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
