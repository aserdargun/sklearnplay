"""Experiment tracking service for sklearn-playground."""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import streamlit as st

from skplay.config import settings


@dataclass
class ExperimentRun:
    """Represents a single experiment run."""

    id: str
    name: str
    task_type: str
    dataset_name: str
    estimator_name: str
    estimator_params: dict[str, Any]
    preprocessing_config: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    primary_metric_name: str | None = None
    training_time_seconds: float | None = None
    code_snippet: str | None = None
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "completed"

    @property
    def primary_metric_value(self) -> float | None:
        """Get the primary metric value."""
        if self.primary_metric_name and self.primary_metric_name in self.metrics:
            return self.metrics[self.primary_metric_name]
        return None


class ExperimentTracker:
    """Service for tracking ML experiments.

    Works with both database-backed storage (Azure) and session-only storage (local).
    """

    def __init__(self):
        self._init_session_storage()

    def _init_session_storage(self) -> None:
        """Initialize session-based experiment storage."""
        if "experiments" not in st.session_state:
            st.session_state.experiments = []
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())

    def _get_user_id(self) -> str | None:
        """Get current user ID if authenticated."""
        from skplay.backend.auth import get_current_user

        user = get_current_user()
        return user.oid if user else None

    def log_experiment(
        self,
        name: str,
        task_type: str,
        dataset_name: str,
        estimator_name: str,
        estimator_params: dict[str, Any],
        preprocessing_config: dict[str, Any],
        metrics: dict[str, float],
        primary_metric_name: str | None = None,
        training_time_seconds: float | None = None,
        code_snippet: str | None = None,
        tags: list[str] | None = None,
    ) -> ExperimentRun:
        """Log a new experiment run.

        Args:
            name: Experiment name
            task_type: Type of ML task
            dataset_name: Name of the dataset used
            estimator_name: Name of the estimator
            estimator_params: Parameters used
            preprocessing_config: Preprocessing configuration
            metrics: Computed metrics
            primary_metric_name: Name of the primary metric
            training_time_seconds: Training duration
            code_snippet: Generated code
            tags: Optional tags

        Returns:
            The created ExperimentRun
        """
        experiment = ExperimentRun(
            id=str(uuid.uuid4()),
            name=name,
            task_type=task_type,
            dataset_name=dataset_name,
            estimator_name=estimator_name,
            estimator_params=estimator_params,
            preprocessing_config=preprocessing_config,
            metrics=metrics,
            primary_metric_name=primary_metric_name,
            training_time_seconds=training_time_seconds,
            code_snippet=code_snippet,
            tags=tags or [],
        )

        # Store in session
        st.session_state.experiments.insert(0, experiment)

        # Store in database if available
        self._persist_to_db(experiment)

        return experiment

    def _persist_to_db(self, experiment: ExperimentRun) -> None:
        """Persist experiment to database if configured."""
        if not settings.enable_experiments:
            return

        try:
            from skplay.backend.database import get_db_context
            from skplay.backend.repositories import ExperimentRepository

            user_id = self._get_user_id()

            with get_db_context() as db:
                repo = ExperimentRepository(db)
                repo.create(
                    name=experiment.name,
                    task_type=experiment.task_type,
                    dataset_name=experiment.dataset_name,
                    estimator_name=experiment.estimator_name,
                    estimator_params=experiment.estimator_params,
                    preprocessing_config=experiment.preprocessing_config,
                    metrics=experiment.metrics,
                    user_id=user_id,
                    session_id=st.session_state.session_id,
                    primary_metric_name=experiment.primary_metric_name,
                    training_time_seconds=experiment.training_time_seconds,
                    code_snippet=experiment.code_snippet,
                    tags=experiment.tags,
                )
        except Exception:
            # Database not configured or error - continue with session storage
            pass

    def get_experiments(
        self,
        task_type: str | None = None,
        limit: int = 50,
        include_db: bool = True,
    ) -> list[ExperimentRun]:
        """Get experiments from session and optionally database.

        Args:
            task_type: Filter by task type
            limit: Maximum number of experiments
            include_db: Whether to include database experiments

        Returns:
            List of experiments
        """
        # Get session experiments
        experiments = list(st.session_state.experiments)

        # Optionally fetch from database
        if include_db and settings.enable_experiments:
            db_experiments = self._fetch_from_db(task_type, limit)
            experiments.extend(db_experiments)

        # Filter and limit
        if task_type:
            experiments = [e for e in experiments if e.task_type == task_type]

        # Sort by created_at descending
        experiments.sort(key=lambda e: e.created_at, reverse=True)

        return experiments[:limit]

    def _fetch_from_db(
        self,
        task_type: str | None,
        limit: int,
    ) -> list[ExperimentRun]:
        """Fetch experiments from database."""
        try:
            from skplay.backend.database import get_db_context
            from skplay.backend.repositories import ExperimentRepository

            user_id = self._get_user_id()

            with get_db_context() as db:
                repo = ExperimentRepository(db)
                db_experiments = repo.list_by_user(
                    user_id=user_id,
                    task_type=task_type,
                    limit=limit,
                )

                return [
                    ExperimentRun(
                        id=e.id,
                        name=e.name,
                        task_type=e.task_type,
                        dataset_name=e.dataset_name,
                        estimator_name=e.estimator_name,
                        estimator_params=e.estimator_params,
                        preprocessing_config=e.preprocessing_config,
                        metrics=e.metrics,
                        primary_metric_name=e.primary_metric_name,
                        training_time_seconds=e.training_time_seconds,
                        code_snippet=e.code_snippet,
                        tags=e.tags,
                        created_at=e.created_at,
                        status=e.status,
                    )
                    for e in db_experiments
                ]
        except Exception:
            return []

    def get_experiment_by_id(self, experiment_id: str) -> ExperimentRun | None:
        """Get a specific experiment by ID."""
        # Check session first
        for exp in st.session_state.experiments:
            if exp.id == experiment_id:
                return exp

        # Check database
        try:
            from skplay.backend.database import get_db_context
            from skplay.backend.repositories import ExperimentRepository

            with get_db_context() as db:
                repo = ExperimentRepository(db)
                e = repo.get_by_id(experiment_id)
                if e:
                    return ExperimentRun(
                        id=e.id,
                        name=e.name,
                        task_type=e.task_type,
                        dataset_name=e.dataset_name,
                        estimator_name=e.estimator_name,
                        estimator_params=e.estimator_params,
                        preprocessing_config=e.preprocessing_config,
                        metrics=e.metrics,
                        primary_metric_name=e.primary_metric_name,
                        training_time_seconds=e.training_time_seconds,
                        code_snippet=e.code_snippet,
                        tags=e.tags,
                        created_at=e.created_at,
                        status=e.status,
                    )
        except Exception:
            pass

        return None

    def clear_session_experiments(self) -> None:
        """Clear all experiments from session storage."""
        st.session_state.experiments = []


class TrainingTimer:
    """Context manager for timing training operations."""

    def __init__(self):
        self.start_time: float | None = None
        self.end_time: float | None = None

    def __enter__(self) -> "TrainingTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.end_time = time.perf_counter()

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.perf_counter()
        return end - self.start_time


def get_experiment_tracker() -> ExperimentTracker:
    """Get the experiment tracker instance."""
    if "experiment_tracker" not in st.session_state:
        st.session_state.experiment_tracker = ExperimentTracker()
    return st.session_state.experiment_tracker
