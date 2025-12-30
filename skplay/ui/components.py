"""Reusable UI components for the Streamlit app."""

import io
from typing import Any

import joblib
import pandas as pd
import streamlit as st

from skplay.core.datasets import DatasetCard, TaskType
from skplay.core.estimators import EstimatorInfo
from skplay.ui.level import Level, get_level, get_level_config, should_show_param


def show_dataset_card(card: DatasetCard) -> None:
    """Display a dataset card with glassmorphism styling.

    Args:
        card: The dataset card to display
    """
    # Header with icon badge
    st.markdown(
        f"""
        <div class="glass-card">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <div class="icon-badge">📊</div>
                <h3 style="margin: 0; margin-left: 0.75rem;">{card.name.replace('_', ' ').title()}</h3>
            </div>
            <p style="color: var(--text-secondary); margin-bottom: 1rem;">{card.description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Samples", card.n_samples)
    with col2:
        st.metric("Features", card.n_features)
    with col3:
        st.metric("Task", card.task_type.replace("_", " ").title())
    with col4:
        st.metric("Domain", card.domain.title())

    # Tags
    if card.tags:
        st.markdown("**Tags:** " + ", ".join(f"`{tag}`" for tag in card.tags))

    # Features (collapsible)
    with st.expander("Feature Details"):
        feature_data = []
        for f in card.features:
            feature_data.append(
                {
                    "Name": f.name,
                    "Type": f.dtype,
                    "Description": f.description[:50] + "..."
                    if len(f.description) > 50
                    else f.description,
                }
            )
        st.dataframe(pd.DataFrame(feature_data), width="stretch", hide_index=True)


def show_data_preview(
    X: pd.DataFrame,
    y: pd.Series | None = None,
    n_rows: int = 5,
) -> None:
    """Show a preview of the data.

    Args:
        X: Feature DataFrame
        y: Target Series (optional)
        n_rows: Number of rows to show
    """
    st.markdown("#### Data Preview")

    # Combine X and y for display
    if y is not None:
        display_df = pd.concat([X, y.rename("target")], axis=1)
    else:
        display_df = X

    st.dataframe(display_df.head(n_rows), width="stretch")

    # Summary stats
    with st.expander("Summary Statistics"):
        st.dataframe(X.describe(), width="stretch")


def show_metrics_table(
    metrics: dict[str, float],
    title: str = "Metrics",
    highlight_best: bool = False,
) -> None:
    """Display metrics in a formatted table with glassmorphism styling.

    Args:
        metrics: Dictionary of metric name -> value
        title: Table title
        highlight_best: Whether to highlight best metric
    """
    st.markdown(
        f"""
        <div class="section-header">
            <div class="icon-badge-sm" style="margin-right: 0.75rem;">📈</div>
            <h4 style="margin: 0;">{title}</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Format metrics for display
    formatted = {}
    for name, value in metrics.items():
        if isinstance(value, float):
            if abs(value) < 0.01 or abs(value) > 1000:
                formatted[name] = f"{value:.4e}"
            else:
                formatted[name] = f"{value:.4f}"
        else:
            formatted[name] = str(value)

    # Display as columns - metrics get glassmorphism from CSS
    cols = st.columns(len(formatted))
    for col, (metric_name, metric_value) in zip(cols, formatted.items(), strict=True):
        col.metric(metric_name.upper().replace("_", " "), metric_value)


def show_parameter_controls(
    estimator_info: EstimatorInfo,
    prefix: str = "",
    key_prefix: str = "",
) -> dict[str, Any]:
    """Render parameter controls for an estimator.

    Args:
        estimator_info: Estimator information
        prefix: Parameter name prefix (for pipeline steps)
        key_prefix: Key prefix for widgets

    Returns:
        Dictionary of parameter name -> value
    """
    level = get_level()
    config = get_level_config()

    # Get default parameters
    try:
        default_est = estimator_info.class_()
        all_params = default_est.get_params(deep=False)
    except Exception:
        all_params = {}

    # Filter params by level
    visible_params = {
        name: value
        for name, value in all_params.items()
        if should_show_param(name, estimator_info.name)
    }

    # Limit number of params shown
    if len(visible_params) > config["max_params_shown"]:
        visible_params = dict(list(visible_params.items())[: config["max_params_shown"]])

    params = {}

    st.markdown("#### Model Parameters")

    if level == "beginner":
        st.info("Showing essential parameters. Increase level for more options.")

    # Key parameters first
    key_params = [p for p in estimator_info.key_params if p in visible_params]
    other_params = [p for p in visible_params if p not in key_params]

    for param_name in key_params + other_params:
        default_value = all_params.get(param_name)
        widget_key = f"{key_prefix}_{param_name}"

        # Determine widget type based on default value
        value = render_param_widget(param_name, default_value, widget_key, estimator_info.name)
        if value is not None:
            params[f"{prefix}{param_name}" if prefix else param_name] = value

    # Tips
    if estimator_info.tips and level == "beginner":
        st.info(f"💡 **Tip:** {estimator_info.tips}")

    return params


def render_param_widget(
    param_name: str,
    default_value: Any,
    key: str,
    estimator_name: str,
) -> Any:
    """Render appropriate widget for a parameter.

    Args:
        param_name: Parameter name
        default_value: Default value
        key: Widget key
        estimator_name: Estimator name (for context)

    Returns:
        Selected value
    """
    # Get help text
    help_text = get_param_help(param_name)

    if default_value is None:
        # Could be int, float, or truly None
        col1, col2 = st.columns([3, 1])
        with col1:
            use_none = st.checkbox(f"{param_name} = None", value=True, key=f"{key}_none")
        if use_none:
            return None
        with col2:
            return st.number_input(param_name, value=0, key=key, help=help_text)

    elif isinstance(default_value, bool):
        return st.checkbox(param_name, value=default_value, key=key, help=help_text)

    elif isinstance(default_value, int):
        # Handle parameters that can be -1 (meaning "auto" or "no limit")
        # e.g., n_jobs=-1 (all cores), max_iter=-1 (no limit), verbose=-1
        if default_value < 0:
            return st.number_input(
                param_name, -1, 10000, default_value, key=key, help=help_text
            )
        # Determine reasonable range for positive values
        elif "n_estimators" in param_name:
            return st.slider(param_name, 10, 1000, default_value, step=10, key=key, help=help_text)
        elif "max_depth" in param_name:
            return st.slider(param_name, 1, 50, default_value or 10, key=key, help=help_text)
        elif "n_neighbors" in param_name or "n_clusters" in param_name:
            return st.slider(param_name, 1, 50, default_value, key=key, help=help_text)
        elif "max_iter" in param_name:
            return st.number_input(param_name, -1, 10000, default_value, key=key, help=help_text)
        elif param_name == "n_jobs":
            # n_jobs: -1 means all cores, None means 1
            return st.number_input(param_name, -1, 32, default_value, key=key, help=help_text)
        elif "n_" in param_name or "min_" in param_name:
            return st.number_input(param_name, 1, 1000, default_value, key=key, help=help_text)
        else:
            return st.number_input(param_name, value=default_value, key=key, help=help_text)

    elif isinstance(default_value, float):
        # Determine reasonable range
        if "learning_rate" in param_name:
            return st.slider(
                param_name, 0.001, 1.0, default_value, step=0.01, key=key, help=help_text
            )
        elif param_name in ("C", "alpha"):
            return st.slider(
                param_name, 0.001, 100.0, float(default_value), key=key, help=help_text
            )
        elif param_name in ("contamination", "subsample"):
            return st.slider(
                param_name, 0.01, 1.0, default_value, step=0.01, key=key, help=help_text
            )
        elif param_name == "eps":
            return st.slider(
                param_name, 0.01, 5.0, default_value, step=0.1, key=key, help=help_text
            )
        elif param_name == "tol":
            return st.number_input(
                param_name, 0.0, 1.0, default_value, format="%.6f", key=key, help=help_text
            )
        else:
            return st.number_input(param_name, value=float(default_value), key=key, help=help_text)

    elif isinstance(default_value, str):
        # Try to get options
        options = get_param_options(param_name, estimator_name)
        if options:
            idx = options.index(default_value) if default_value in options else 0
            return st.selectbox(param_name, options, index=idx, key=key, help=help_text)
        else:
            return st.text_input(param_name, default_value, key=key, help=help_text)

    elif isinstance(default_value, tuple):
        # For things like hidden_layer_sizes
        value_str = st.text_input(
            param_name,
            str(default_value),
            key=key,
            help=help_text + " (enter as tuple, e.g., (100, 50))",
        )
        try:
            return eval(value_str)
        except Exception:
            return default_value

    else:
        st.text(f"{param_name}: {default_value} (type: {type(default_value).__name__})")
        return default_value


def get_param_options(param_name: str, estimator_name: str) -> list[str | None] | None:
    """Get valid options for a string parameter.

    Args:
        param_name: Parameter name
        estimator_name: Estimator name

    Returns:
        List of options or None
    """
    # Handle estimator-specific options
    estimator_lower = estimator_name.lower()

    # Criterion depends on estimator type
    if param_name == "criterion":
        if "gradientboosting" in estimator_lower or "histgradientboosting" in estimator_lower:
            return ["friedman_mse", "squared_error"]
        elif "regressor" in estimator_lower or "regression" in estimator_lower:
            return ["squared_error", "friedman_mse", "absolute_error", "poisson"]
        else:  # Classifiers (DecisionTree, RandomForest, etc.)
            return ["gini", "entropy", "log_loss"]

    # Loss depends on estimator type
    if param_name == "loss":
        if "sgdclassifier" in estimator_lower:
            return ["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"]
        elif "sgdregressor" in estimator_lower:
            return ["squared_error", "huber", "epsilon_insensitive"]
        elif "gradientboostingclassifier" in estimator_lower:
            return ["log_loss", "exponential"]
        elif "gradientboostingregressor" in estimator_lower:
            return ["squared_error", "absolute_error", "huber", "quantile"]

    options_map: dict[str, list[str | None]] = {
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "solver": [
            "auto",
            "svd",
            "cholesky",
            "lsqr",
            "sparse_cg",
            "sag",
            "saga",
            "lbfgs",
            "liblinear",
            "newton-cg",
        ],
        "penalty": [None, "l1", "l2", "elasticnet"],
        "max_features": [None, "sqrt", "log2"],
        "init": ["k-means++", "random"],
        "linkage": ["ward", "complete", "average", "single"],
        "metric": ["euclidean", "manhattan", "cosine", "minkowski"],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "activation": ["identity", "logistic", "tanh", "relu"],
        "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
        "covariance_type": ["full", "tied", "diag", "spherical"],
        "multi_class": ["auto", "ovr", "multinomial"],
    }

    return options_map.get(param_name)


def get_param_help(param_name: str) -> str:
    """Get help text for a parameter.

    Args:
        param_name: Parameter name

    Returns:
        Help text
    """
    help_texts = {
        "n_estimators": "Number of trees/estimators in the ensemble",
        "max_depth": "Maximum depth of trees. Limit to prevent overfitting.",
        "learning_rate": "Step size for gradient updates. Lower = slower but often better.",
        "C": "Regularization strength (inverse). Higher = less regularization.",
        "alpha": "Regularization strength. Higher = more regularization.",
        "gamma": "Kernel coefficient. 'scale' uses 1/(n_features * X.var()).",
        "n_neighbors": "Number of neighbors to consider.",
        "n_clusters": "Number of clusters to form.",
        "min_samples_split": "Minimum samples required to split an internal node.",
        "min_samples_leaf": "Minimum samples required to be at a leaf node.",
        "max_features": "Number of features to consider for splits.",
        "kernel": "Kernel type to be used in the algorithm.",
        "solver": "Algorithm to use in the optimization problem.",
        "penalty": "Type of regularization penalty.",
        "criterion": "Function to measure the quality of a split.",
        "max_iter": "Maximum number of iterations for the solver.",
        "tol": "Tolerance for stopping criterion.",
        "subsample": "Fraction of samples used for fitting trees.",
        "contamination": "Expected proportion of outliers in the data.",
        "eps": "Maximum distance between samples for neighborhood.",
        "min_samples": "Minimum samples in a neighborhood for core points.",
    }

    return help_texts.get(param_name, "")


def show_code_snippet(code: str, language: str = "python") -> None:
    """Display a code snippet with glassmorphism styling.

    Args:
        code: The code to display
        language: Programming language
    """
    st.markdown(
        """
        <div class="section-header">
            <div class="icon-badge-sm" style="margin-right: 0.75rem;">💻</div>
            <h4 style="margin: 0;">Reproducible Code</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code(code, language=language)


def create_download_button(
    model: Any,
    filename: str = "model.joblib",
) -> None:
    """Create download button for a trained model with glassmorphism styling.

    Args:
        model: The trained model/pipeline
        filename: Download filename
    """
    st.markdown(
        """
        <div class="section-header">
            <div class="icon-badge-sm" style="margin-right: 0.75rem;">💾</div>
            <h4 style="margin: 0;">Export Model</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        "**Security Warning:** Only load models from trusted sources. "
        "Malicious models can execute arbitrary code."
    )

    # Serialize model to bytes
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)

    st.download_button(
        label="📥 Download Model (joblib)",
        data=buffer,
        file_name=filename,
        mime="application/octet-stream",
    )


def show_estimator_selector(
    task_type: TaskType,
    level: Level,
    key: str = "estimator_select",
) -> EstimatorInfo | None:
    """Show estimator selection widget.

    Args:
        task_type: The task type
        level: User level
        key: Widget key

    Returns:
        Selected estimator info or None
    """
    from skplay.core.estimators import get_default_estimator, get_estimators_for_task

    estimators = get_estimators_for_task(task_type, level)

    if not estimators:
        st.warning(f"No estimators available for {task_type}")
        return None

    # Get default
    default_name = get_default_estimator(task_type)
    names = [e.name for e in estimators]
    default_idx = names.index(default_name) if default_name in names else 0

    selected_name = st.selectbox(
        "Select Model",
        options=names,
        index=default_idx,
        key=key,
    )

    selected = next((e for e in estimators if e.name == selected_name), None)

    if selected:
        st.markdown(f"*{selected.description}*")
        if selected.doc_url:
            st.markdown(f"[📚 Documentation]({selected.doc_url})")

    return selected


def show_split_controls(
    n_samples: int,
    key_prefix: str = "split",
) -> dict:
    """Show train/test split controls.

    Args:
        n_samples: Number of samples
        key_prefix: Key prefix for widgets

    Returns:
        Split configuration dictionary
    """
    config = get_level_config()

    st.markdown("#### Data Split")

    col1, col2 = st.columns(2)

    with col1:
        test_size = st.slider(
            "Test Set Size",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            key=f"{key_prefix}_test_size",
            help="Proportion of data to use for testing",
        )

    with col2:
        random_state = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=9999,
            value=42,
            key=f"{key_prefix}_random_state",
            help="For reproducibility",
        )

    result = {
        "test_size": test_size,
        "random_state": random_state,
    }

    # CV options for intermediate+
    if config["show_cv"]:
        use_cv = st.checkbox(
            "Use Cross-Validation",
            value=False,
            key=f"{key_prefix}_use_cv",
        )
        if use_cv:
            result["cv_folds"] = st.slider(
                "CV Folds",
                min_value=2,
                max_value=10,
                value=config["default_cv_folds"],
                key=f"{key_prefix}_cv_folds",
            )
        result["use_cv"] = use_cv

    # Show split info
    train_size = int(n_samples * (1 - test_size))
    test_size_n = n_samples - train_size
    st.caption(f"Training: {train_size} samples | Test: {test_size_n} samples")

    return result


def show_preprocessing_controls(
    numeric_cols: list[str],
    categorical_cols: list[str],
    task_type: str,
    key_prefix: str = "preproc",
) -> dict:
    """Show preprocessing configuration controls.

    Args:
        numeric_cols: Numeric column names
        categorical_cols: Categorical column names
        task_type: The task type
        key_prefix: Key prefix for widgets

    Returns:
        Preprocessing configuration dictionary
    """
    from skplay.core.preprocessing import get_preprocessing_options

    level = get_level()
    options = get_preprocessing_options(level)

    config = {}

    st.markdown("#### Preprocessing")

    col1, col2 = st.columns(2)

    # Numeric preprocessing
    with col1:
        st.markdown("**Numeric Features**")

        if numeric_cols:
            config["numeric_imputer"] = st.selectbox(
                "Imputation",
                options=options["numeric_imputers"],
                index=options["numeric_imputers"].index("median")
                if "median" in options["numeric_imputers"]
                else 0,
                key=f"{key_prefix}_num_imputer",
                help="Strategy for handling missing numeric values",
            )

            config["numeric_scaler"] = st.selectbox(
                "Scaling",
                options=options["numeric_scalers"],
                index=options["numeric_scalers"].index("standard")
                if "standard" in options["numeric_scalers"]
                else 0,
                key=f"{key_prefix}_num_scaler",
                help="Feature scaling method",
            )
        else:
            st.caption("No numeric columns")

    # Categorical preprocessing
    with col2:
        st.markdown("**Categorical Features**")

        if categorical_cols:
            config["categorical_imputer"] = st.selectbox(
                "Imputation",
                options=options["categorical_imputers"],
                index=0,
                key=f"{key_prefix}_cat_imputer",
            )

            config["categorical_encoder"] = st.selectbox(
                "Encoding",
                options=options["categorical_encoders"],
                index=0,
                key=f"{key_prefix}_cat_encoder",
            )
        else:
            st.caption("No categorical columns")

    # Advanced options
    level_config = get_level_config()

    if level_config["show_feature_selection"]:
        with st.expander("Feature Selection"):
            config["feature_selector"] = st.selectbox(
                "Method",
                options=options["feature_selectors"],
                key=f"{key_prefix}_feat_select",
            )

            if config["feature_selector"] != "none":
                if "kbest" in config["feature_selector"]:
                    config["feature_selector_k"] = st.slider(
                        "Number of Features",
                        min_value=1,
                        max_value=max(len(numeric_cols) + len(categorical_cols), 1),
                        value=min(10, len(numeric_cols) + len(categorical_cols)),
                        key=f"{key_prefix}_k_features",
                    )
                elif "percentile" in config["feature_selector"]:
                    config["feature_selector_percentile"] = st.slider(
                        "Percentile",
                        min_value=10,
                        max_value=100,
                        value=50,
                        key=f"{key_prefix}_percentile",
                    )

    if level_config["show_dim_reduction"]:
        with st.expander("Dimensionality Reduction"):
            config["dim_reducer"] = st.selectbox(
                "Method",
                options=options["dim_reducers"],
                key=f"{key_prefix}_dim_red",
            )

            if config["dim_reducer"] != "none":
                config["dim_reducer_n_components"] = st.slider(
                    "Components / Variance Ratio",
                    min_value=0.5,
                    max_value=0.99,
                    value=0.95,
                    key=f"{key_prefix}_n_components",
                    help="Number of components or variance ratio to keep",
                )

    return config


def show_training_button(key: str = "train_btn") -> bool:
    """Show training button with modern styling.

    Args:
        key: Widget key

    Returns:
        True if button clicked
    """
    return st.button(
        "🚀 Train Model",
        key=key,
        type="primary",
        width="stretch",
    )


def show_progress_message(message: str) -> None:
    """Show a progress message.

    Args:
        message: Message to display
    """
    st.info(f"⏳ {message}")
