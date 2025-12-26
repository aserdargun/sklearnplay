"""Unsupervised Learning - Clustering, Dimensionality Reduction, Outlier Detection.

This page covers:
- Clustering (K-Means, DBSCAN, Hierarchical, etc.)
- Dimensionality Reduction (PCA, t-SNE, etc.)
- Outlier Detection (Isolation Forest, LOF, etc.)
"""

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Unsupervised Learning", page_icon="🔍", layout="wide")

from skplay.core.datasets import DatasetRegistry, get_dataset
from skplay.core.estimators import create_estimator
from skplay.core.evaluation import (
    compute_clustering_metrics,
    compute_outlier_metrics,
    plot_cluster_visualization,
    plot_outlier_scores,
)
from skplay.core.preprocessing import PreprocessingBuilder, identify_column_types
from skplay.core.snippets import generate_code_snippet
from skplay.ui.components import (
    show_code_snippet,
    show_data_preview,
    show_dataset_card,
    show_estimator_selector,
    show_metrics_table,
    show_parameter_controls,
    show_preprocessing_controls,
    show_training_button,
)
from skplay.ui.level import get_level, get_level_config, level_selector


def main():
    st.title("🔍 Unsupervised Learning")

    st.markdown("""
    Unsupervised learning finds patterns in data without labeled outcomes.

    **Clustering** groups similar data points together.

    **Dimensionality Reduction** projects high-dimensional data to lower dimensions.

    **Outlier Detection** identifies unusual data points.

    [📚 sklearn User Guide: Unsupervised Learning](https://scikit-learn.org/stable/unsupervised_learning.html)
    """)

    with st.sidebar:
        level_selector()
        st.markdown("---")

    # Task type selection
    task_type = st.radio(
        "Task",
        options=["clustering", "outlier_detection"],
        format_func=lambda x: x.replace("_", " ").title(),
        horizontal=True,
        key="unsup_task",
    )

    tabs = st.tabs(["📊 Data", "🔧 Preprocessing", "🤖 Model", "📈 Results", "💡 Learn More"])

    with tabs[0]:
        data_result = data_section(task_type)

    with tabs[1]:
        if data_result:
            preproc_config = preprocessing_section(data_result)
        else:
            st.info("👆 Select a dataset first")
            preproc_config = None

    with tabs[2]:
        if data_result and preproc_config is not None:
            model_result = model_section(data_result, preproc_config, task_type)
        else:
            st.info("👆 Configure preprocessing first")
            model_result = None

    with tabs[3]:
        if model_result:
            results_section(model_result, data_result, task_type)
        else:
            st.info("👆 Train a model first")

    with tabs[4]:
        learn_more_section(task_type)


@st.cache_data
def load_dataset(name: str):
    """Load and cache a dataset."""
    return get_dataset(name)


def data_section(task_type):
    """Dataset selection section."""
    st.header("Select Dataset")

    # Dataset source
    data_source = st.radio(
        "Data Source",
        options=["toy_dataset", "upload"],
        format_func=lambda x: "Toy Dataset" if x == "toy_dataset" else "Upload CSV",
        horizontal=True,
        key="unsup_data_source",
    )

    if data_source == "toy_dataset":
        # Get datasets - clustering has no target, outlier detection may have labels
        if task_type == "clustering":
            available = DatasetRegistry.list_by_task("clustering")
            # Also include any dataset without target
            for name in DatasetRegistry.list_all():
                if name not in available:
                    available.append(name)
        else:
            available = DatasetRegistry.list_by_task("outlier_detection")
            # Add classification datasets for demo
            available.extend(DatasetRegistry.list_by_task("classification"))

        available = list(set(available))

        if not available:
            st.warning(f"No datasets available for {task_type}")
            return None

        selected_dataset = st.selectbox(
            "Dataset",
            options=available,
            key="unsup_selected_dataset",
        )

        if selected_dataset:
            result = load_dataset(selected_dataset)
            show_dataset_card(result.card)
            show_data_preview(result.X, result.y)

            st.session_state.unsup_current_data = result
            return result

    else:
        # CSV Upload
        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            key="unsup_csv_upload",
        )

        if uploaded_file:
            from skplay.core.upload import create_dataset_from_upload

            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(), use_container_width=True)

            # Optional target for evaluation
            target_col = st.selectbox(
                "Target Column (optional, for evaluation)",
                options=[None] + list(df.columns),
                format_func=lambda x: "(No target)" if x is None else x,
                key="unsup_target_col",
            )

            if st.button("Create Dataset", key="unsup_create"):
                result = create_dataset_from_upload(
                    df,
                    target_column=target_col,
                    task_type=task_type,
                    dataset_name=uploaded_file.name.replace(".csv", ""),
                )
                st.session_state.unsup_current_data = result
                st.success("Dataset created!")
                return result

    return st.session_state.get("unsup_current_data")


def preprocessing_section(data_result):
    """Preprocessing configuration section."""
    st.header("Configure Preprocessing")

    X = data_result.X
    task_type = data_result.card.task_type

    numeric_cols, categorical_cols = identify_column_types(X)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Numeric columns:** {len(numeric_cols)}")
    with col2:
        st.markdown(f"**Categorical columns:** {len(categorical_cols)}")

    config = show_preprocessing_controls(
        numeric_cols, categorical_cols, task_type, key_prefix="unsup_preproc"
    )

    config["numeric_cols"] = numeric_cols
    config["categorical_cols"] = categorical_cols
    st.session_state.unsup_preproc_config = config

    return config


def model_section(data_result, preproc_config, task_type):
    """Model selection and training section."""
    st.header("Select and Train Model")

    level = get_level()

    # Estimator selection
    estimator_info = show_estimator_selector(task_type, level, key="unsup_estimator")

    if not estimator_info:
        return None

    st.markdown("---")

    # Parameter controls
    params = show_parameter_controls(estimator_info, prefix="", key_prefix="unsup_params")

    st.markdown("---")

    if show_training_button(key="unsup_train"):
        with st.spinner("Training model..."):
            result = train_unsupervised_model(
                data_result, preproc_config, estimator_info, params, task_type
            )
            st.session_state.unsup_training_result = result
            st.success("Training complete!")
            return result

    return st.session_state.get("unsup_training_result")


def train_unsupervised_model(data_result, preproc_config, estimator_info, params, task_type):
    """Train an unsupervised model."""
    X = data_result.X
    y = data_result.y  # May be None or used for evaluation

    # Build preprocessing pipeline
    preproc_builder = PreprocessingBuilder(
        numeric_imputer=preproc_config.get("numeric_imputer", "median"),
        numeric_scaler=preproc_config.get("numeric_scaler", "standard"),
        categorical_imputer=preproc_config.get("categorical_imputer", "most_frequent"),
        categorical_encoder=preproc_config.get("categorical_encoder", "onehot"),
        task_type=task_type,
    )

    preprocessing_pipeline, numeric_cols, categorical_cols = preproc_builder.build_full_pipeline(X)

    # Transform data
    X_transformed = preprocessing_pipeline.fit_transform(X)

    # Create and fit estimator
    estimator = create_estimator(task_type, estimator_info.name, **params)

    if task_type == "clustering":
        # Clustering
        if hasattr(estimator, "fit_predict"):
            labels = estimator.fit_predict(X_transformed)
        else:
            estimator.fit(X_transformed)
            labels = estimator.labels_

        metrics = compute_clustering_metrics(X_transformed, labels)

        return {
            "estimator": estimator,
            "preprocessing": preprocessing_pipeline,
            "X_transformed": X_transformed,
            "labels": labels,
            "metrics": metrics,
            "y_true": y,
            "estimator_info": estimator_info,
            "params": params,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
        }

    else:
        # Outlier detection
        predictions = estimator.fit_predict(X_transformed)

        scores = None
        if hasattr(estimator, "decision_function"):
            scores = estimator.decision_function(X_transformed)
        elif hasattr(estimator, "score_samples"):
            scores = estimator.score_samples(X_transformed)

        metrics = {}
        if y is not None:
            # Convert labels if needed
            if y.dtype == object:
                positive_labels = {"fraud", "outlier", "anomaly", "1", "true", "yes"}
                y_binary = np.array([1 if str(v).lower() in positive_labels else 0 for v in y])
            else:
                y_binary = y.values

            metrics = compute_outlier_metrics(y_binary, predictions, scores)

        return {
            "estimator": estimator,
            "preprocessing": preprocessing_pipeline,
            "X_transformed": X_transformed,
            "predictions": predictions,
            "scores": scores,
            "metrics": metrics,
            "y_true": y,
            "estimator_info": estimator_info,
            "params": params,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
        }


def results_section(model_result, data_result, task_type):
    """Display results and visualizations."""
    st.header("Results")

    config = get_level_config()

    if task_type == "clustering":
        # Clustering metrics
        metrics = model_result["metrics"]
        show_metrics_table(metrics, "Clustering Metrics")

        st.markdown("---")

        # Visualization
        st.subheader("Cluster Visualization")

        viz_method = st.radio(
            "Visualization Method",
            options=["pca", "first_two"],
            format_func=lambda x: "PCA" if x == "pca" else "First Two Features",
            horizontal=True,
            key="cluster_viz_method",
        )

        fig = plot_cluster_visualization(
            model_result["X_transformed"],
            model_result["labels"],
            method=viz_method,
        )
        st.pyplot(fig)

        # Cluster summary
        st.markdown("#### Cluster Summary")
        labels = model_result["labels"]
        unique_labels = np.unique(labels)

        summary_data = []
        for label in unique_labels:
            mask = labels == label
            summary_data.append(
                {
                    "Cluster": label if label >= 0 else "Noise",
                    "Count": mask.sum(),
                    "Percentage": f"{mask.mean() * 100:.1f}%",
                }
            )

        st.dataframe(pd.DataFrame(summary_data), hide_index=True)

    else:
        # Outlier detection
        if model_result["metrics"]:
            show_metrics_table(model_result["metrics"], "Detection Metrics")
        else:
            st.info("No ground truth labels available for metrics")

        st.markdown("---")

        # Score distribution
        st.subheader("Score Distribution")

        if model_result["scores"] is not None:
            fig = plot_outlier_scores(
                model_result["scores"],
                y_true=model_result["y_true"].values
                if model_result["y_true"] is not None
                else None,
            )
            st.pyplot(fig)

        # Detection summary
        st.markdown("#### Detection Summary")
        predictions = model_result["predictions"]
        n_outliers = (predictions == -1).sum()
        n_normal = (predictions == 1).sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Normal", n_normal)
        col2.metric("Outliers", n_outliers)
        col3.metric("Outlier %", f"{n_outliers / len(predictions) * 100:.1f}%")

    # Code snippet
    if config["show_code_snippet"]:
        st.markdown("---")
        st.subheader("Reproducible Code")

        # Create a simple pipeline for code generation
        from sklearn.pipeline import Pipeline as SKPipeline

        simple_pipeline = SKPipeline(
            [
                ("preprocessing", model_result["preprocessing"]),
                ("estimator", model_result["estimator"]),
            ]
        )

        code = generate_code_snippet(
            simple_pipeline,
            X_train_shape=data_result.X.shape,
            y_train_shape=None,
            dataset_name=data_result.card.name,
            task_type=task_type,
            numeric_cols=model_result["numeric_cols"],
            categorical_cols=model_result["categorical_cols"],
        )

        show_code_snippet(code)


def learn_more_section(task_type):
    """Educational content about unsupervised learning."""
    st.header("Learn More")

    level = get_level()

    if task_type == "clustering":
        st.markdown("""
        ### Clustering

        Clustering groups similar data points without predefined labels.
        The goal is to maximize similarity within clusters and dissimilarity between clusters.
        """)

        with st.expander("K-Means"):
            st.markdown("""
            **How it works:**
            1. Initialize K cluster centers randomly
            2. Assign each point to nearest center
            3. Recompute centers as cluster means
            4. Repeat until convergence

            **Pros:** Fast, simple, works well on spherical clusters

            **Cons:** Must specify K, sensitive to initialization, struggles with non-spherical clusters

            **Key parameter:** `n_clusters` - number of clusters to form
            """)

        with st.expander("DBSCAN"):
            st.markdown("""
            **How it works:**
            1. For each point, find neighbors within eps distance
            2. Points with >= min_samples neighbors are core points
            3. Connect core points to form clusters
            4. Non-core points near clusters are border points
            5. Remaining points are noise

            **Pros:** Finds arbitrary shapes, handles noise, doesn't need K

            **Cons:** Sensitive to eps and min_samples, struggles with varying densities

            **Key parameters:** `eps` (radius), `min_samples` (density threshold)
            """)

        with st.expander("Hierarchical Clustering"):
            st.markdown("""
            Builds a tree of clusters (dendrogram).

            **Agglomerative (bottom-up):**
            1. Start with each point as its own cluster
            2. Merge closest clusters repeatedly
            3. Stop when reaching K clusters

            **Linkage methods:**
            - Ward: Minimizes variance
            - Complete: Maximum distance between clusters
            - Average: Average distance between clusters
            """)

    else:  # outlier_detection
        st.markdown("""
        ### Outlier Detection

        Identifies data points that differ significantly from the majority.
        Useful for fraud detection, quality control, and data cleaning.
        """)

        with st.expander("Isolation Forest"):
            st.markdown("""
            **Intuition:** Outliers are easier to isolate than normal points.

            **How it works:**
            1. Build random trees that randomly split features
            2. Outliers require fewer splits to isolate
            3. Score based on average path length

            **Pros:** Fast, handles high dimensions well

            **Key parameter:** `contamination` - expected proportion of outliers
            """)

        with st.expander("Local Outlier Factor (LOF)"):
            st.markdown("""
            **Intuition:** Outliers have lower local density than neighbors.

            **How it works:**
            1. Compute local density for each point
            2. Compare to neighbors' densities
            3. Points much less dense than neighbors are outliers

            **Pros:** Detects local outliers, not just global

            **Key parameter:** `n_neighbors` - neighborhood size
            """)

    if level in ("intermediate", "advanced"):
        with st.expander("Evaluation Metrics"):
            if task_type == "clustering":
                st.markdown("""
                **Silhouette Score:** Measures cluster cohesion and separation (-1 to 1, higher is better)

                **Calinski-Harabasz:** Ratio of between-cluster to within-cluster variance (higher is better)

                **Davies-Bouldin:** Average similarity between clusters (lower is better)

                Note: These metrics don't require ground truth labels.
                """)
            else:
                st.markdown("""
                **When labels are available:**
                - Precision: Fraction of detected outliers that are true outliers
                - Recall: Fraction of true outliers that are detected
                - F1 Score: Harmonic mean of precision and recall
                - ROC AUC: Area under ROC curve

                **Without labels:**
                - Examine score distributions
                - Domain expertise for threshold selection
                """)


if __name__ == "__main__":
    main()
