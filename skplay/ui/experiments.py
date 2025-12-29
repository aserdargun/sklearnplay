"""Experiment tracking UI components for Streamlit."""

import pandas as pd
import streamlit as st

from skplay.backend.experiments import ExperimentRun, get_experiment_tracker


def show_experiment_history(
    task_type: str | None = None,
    limit: int = 20,
    show_filters: bool = True,
) -> None:
    """Display experiment history with comparison capabilities.

    Args:
        task_type: Filter by task type
        limit: Maximum experiments to show
        show_filters: Whether to show filter controls
    """
    tracker = get_experiment_tracker()

    st.markdown("### Experiment History")

    # Filters
    if show_filters:
        col1, col2 = st.columns([2, 1])
        with col1:
            task_filter = st.selectbox(
                "Task Type",
                options=["All", "classification", "regression", "clustering", "outlier_detection"],
                key="exp_task_filter",
            )
            if task_filter != "All":
                task_type = task_filter
        with col2:
            limit = st.slider("Show", 5, 50, limit, key="exp_limit")

    experiments = tracker.get_experiments(task_type=task_type, limit=limit)

    if not experiments:
        st.info("No experiments recorded yet. Train a model to see it here!")
        return

    # Convert to DataFrame for display
    exp_data = []
    for exp in experiments:
        row = {
            "Name": exp.name,
            "Model": exp.estimator_name,
            "Dataset": exp.dataset_name,
            "Task": exp.task_type,
        }

        # Add primary metric
        if exp.primary_metric_name and exp.primary_metric_value is not None:
            row[exp.primary_metric_name.upper()] = f"{exp.primary_metric_value:.4f}"

        # Add training time
        if exp.training_time_seconds:
            row["Time (s)"] = f"{exp.training_time_seconds:.2f}"

        row["Created"] = exp.created_at.strftime("%Y-%m-%d %H:%M")
        row["_id"] = exp.id  # Hidden column for selection

        exp_data.append(row)

    df = pd.DataFrame(exp_data)

    # Display table
    st.dataframe(
        df.drop(columns=["_id"]),
        use_container_width=True,
        hide_index=True,
    )

    # Experiment details expander
    with st.expander("View Experiment Details"):
        selected_exp_name = st.selectbox(
            "Select Experiment",
            options=[e.name for e in experiments],
            key="exp_detail_select",
        )

        selected_exp = next((e for e in experiments if e.name == selected_exp_name), None)

        if selected_exp:
            _show_experiment_details(selected_exp)


def _show_experiment_details(experiment: ExperimentRun) -> None:
    """Show detailed view of a single experiment."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Configuration**")
        st.json({
            "Estimator": experiment.estimator_name,
            "Parameters": experiment.estimator_params,
            "Preprocessing": experiment.preprocessing_config,
        })

    with col2:
        st.markdown("**Metrics**")
        for metric_name, value in experiment.metrics.items():
            st.metric(
                metric_name.upper(),
                f"{value:.4f}" if isinstance(value, float) else str(value),
            )

    # Code snippet
    if experiment.code_snippet:
        st.markdown("**Generated Code**")
        st.code(experiment.code_snippet, language="python")

    # Tags
    if experiment.tags:
        st.markdown("**Tags:** " + ", ".join(f"`{tag}`" for tag in experiment.tags))


def show_experiment_comparison(experiments: list[ExperimentRun]) -> None:
    """Show side-by-side comparison of experiments.

    Args:
        experiments: List of experiments to compare (max 4)
    """
    if len(experiments) < 2:
        st.warning("Select at least 2 experiments to compare")
        return

    experiments = experiments[:4]  # Limit to 4

    st.markdown("### Experiment Comparison")

    # Create columns for each experiment
    cols = st.columns(len(experiments))

    for col, exp in zip(cols, experiments, strict=False):
        with col:
            st.markdown(f"**{exp.name}**")
            st.caption(f"{exp.estimator_name}")

            # Metrics
            for metric_name, value in exp.metrics.items():
                st.metric(
                    metric_name,
                    f"{value:.4f}" if isinstance(value, float) else str(value),
                )

            # Key params
            with st.expander("Parameters"):
                st.json(exp.estimator_params)

    # Metrics comparison chart
    _show_metrics_comparison_chart(experiments)


def _show_metrics_comparison_chart(experiments: list[ExperimentRun]) -> None:
    """Show metrics comparison as a bar chart."""
    import plotly.express as px

    # Collect all metrics
    all_metrics = set()
    for exp in experiments:
        all_metrics.update(exp.metrics.keys())

    if not all_metrics:
        return

    # Build data for chart
    chart_data = []
    for exp in experiments:
        for metric_name in all_metrics:
            if metric_name in exp.metrics:
                chart_data.append({
                    "Experiment": exp.name,
                    "Metric": metric_name.upper(),
                    "Value": exp.metrics[metric_name],
                })

    if not chart_data:
        return

    df = pd.DataFrame(chart_data)

    fig = px.bar(
        df,
        x="Metric",
        y="Value",
        color="Experiment",
        barmode="group",
        title="Metrics Comparison",
    )

    st.plotly_chart(fig, use_container_width=True)


def show_quick_experiment_card(experiment: ExperimentRun) -> None:
    """Show a compact card for a single experiment."""
    with st.container():
        st.markdown(f"**{experiment.name}**")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"Model: {experiment.estimator_name}")
        with col2:
            if experiment.primary_metric_value is not None:
                st.caption(
                    f"{experiment.primary_metric_name}: {experiment.primary_metric_value:.4f}"
                )
        with col3:
            if experiment.training_time_seconds:
                st.caption(f"Time: {experiment.training_time_seconds:.2f}s")


def experiment_logging_widget(
    task_type: str,
    dataset_name: str,
    estimator_name: str,
    estimator_params: dict,
    preprocessing_config: dict,
    metrics: dict,
    training_time_seconds: float | None = None,
    code_snippet: str | None = None,
    key_prefix: str = "exp_log",
) -> ExperimentRun | None:
    """Widget to log the current training run as an experiment.

    Returns:
        The logged experiment if saved, None otherwise
    """
    tracker = get_experiment_tracker()

    with st.expander("Save Experiment", expanded=False):
        # Auto-generate name
        default_name = f"{estimator_name}_{dataset_name}"

        exp_name = st.text_input(
            "Experiment Name",
            value=default_name,
            key=f"{key_prefix}_name",
        )

        # Tags
        tags_input = st.text_input(
            "Tags (comma-separated)",
            placeholder="e.g., baseline, v1, tuned",
            key=f"{key_prefix}_tags",
        )
        tags = [t.strip() for t in tags_input.split(",") if t.strip()]

        # Primary metric selection
        metric_names = list(metrics.keys())
        primary_metric = st.selectbox(
            "Primary Metric",
            options=metric_names,
            index=0 if metric_names else None,
            key=f"{key_prefix}_primary_metric",
        )

        if st.button("Save Experiment", key=f"{key_prefix}_save_btn", type="primary"):
            experiment = tracker.log_experiment(
                name=exp_name,
                task_type=task_type,
                dataset_name=dataset_name,
                estimator_name=estimator_name,
                estimator_params=estimator_params,
                preprocessing_config=preprocessing_config,
                metrics=metrics,
                primary_metric_name=primary_metric,
                training_time_seconds=training_time_seconds,
                code_snippet=code_snippet,
                tags=tags,
            )
            st.success(f"Experiment '{exp_name}' saved!")
            return experiment

    return None
