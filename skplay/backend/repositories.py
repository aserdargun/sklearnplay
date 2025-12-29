"""Repository pattern for database operations."""

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from skplay.backend.models import Experiment, Model, User


class UserRepository:
    """Repository for User operations."""

    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, user_id: str) -> User | None:
        """Get user by ID."""
        return self.db.get(User, user_id)

    def get_by_azure_oid(self, azure_oid: str) -> User | None:
        """Get user by Azure Object ID."""
        stmt = select(User).where(User.azure_oid == azure_oid)
        return self.db.scalar(stmt)

    def get_by_email(self, email: str) -> User | None:
        """Get user by email."""
        stmt = select(User).where(User.email == email)
        return self.db.scalar(stmt)

    def create(
        self,
        email: str,
        display_name: str,
        azure_oid: str | None = None,
        preferred_level: str = "beginner",
    ) -> User:
        """Create a new user."""
        user = User(
            email=email,
            display_name=display_name,
            azure_oid=azure_oid,
            preferred_level=preferred_level,
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def get_or_create_from_azure(
        self,
        azure_oid: str,
        email: str,
        display_name: str,
    ) -> tuple[User, bool]:
        """Get existing user or create new one from Azure Entra ID claims.

        Returns:
            Tuple of (user, created) where created is True if user was newly created.
        """
        user = self.get_by_azure_oid(azure_oid)
        if user:
            # Update last login
            user.last_login = datetime.utcnow()
            self.db.commit()
            return user, False

        # Create new user
        user = self.create(
            email=email,
            display_name=display_name,
            azure_oid=azure_oid,
        )
        return user, True

    def update_level(self, user_id: str, level: str) -> User | None:
        """Update user's preferred level."""
        user = self.get_by_id(user_id)
        if user:
            user.preferred_level = level
            self.db.commit()
            self.db.refresh(user)
        return user


class ExperimentRepository:
    """Repository for Experiment operations."""

    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, experiment_id: str) -> Experiment | None:
        """Get experiment by ID."""
        return self.db.get(Experiment, experiment_id)

    def list_by_user(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        task_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[Experiment]:
        """List experiments with optional filters."""
        stmt = select(Experiment).order_by(desc(Experiment.created_at))

        if user_id:
            stmt = stmt.where(Experiment.user_id == user_id)
        if session_id:
            stmt = stmt.where(Experiment.session_id == session_id)
        if task_type:
            stmt = stmt.where(Experiment.task_type == task_type)

        stmt = stmt.limit(limit).offset(offset)
        return self.db.scalars(stmt).all()

    def create(
        self,
        name: str,
        task_type: str,
        dataset_name: str,
        estimator_name: str,
        estimator_params: dict[str, Any],
        preprocessing_config: dict[str, Any],
        metrics: dict[str, float],
        user_id: str | None = None,
        session_id: str | None = None,
        primary_metric_name: str | None = None,
        training_time_seconds: float | None = None,
        code_snippet: str | None = None,
        tags: list[str] | None = None,
    ) -> Experiment:
        """Create a new experiment."""
        # Determine primary metric value
        primary_metric_value = None
        if primary_metric_name and primary_metric_name in metrics:
            primary_metric_value = metrics[primary_metric_name]

        experiment = Experiment(
            name=name,
            task_type=task_type,
            dataset_name=dataset_name,
            estimator_name=estimator_name,
            estimator_params=estimator_params,
            preprocessing_config=preprocessing_config,
            metrics=metrics,
            user_id=user_id,
            session_id=session_id,
            primary_metric_name=primary_metric_name,
            primary_metric_value=primary_metric_value,
            training_time_seconds=training_time_seconds,
            code_snippet=code_snippet,
            tags=tags or [],
            status="completed",
            completed_at=datetime.utcnow(),
        )
        self.db.add(experiment)
        self.db.commit()
        self.db.refresh(experiment)
        return experiment

    def delete(self, experiment_id: str) -> bool:
        """Delete an experiment."""
        experiment = self.get_by_id(experiment_id)
        if experiment:
            self.db.delete(experiment)
            self.db.commit()
            return True
        return False

    def get_best_by_metric(
        self,
        task_type: str,
        metric_name: str,
        user_id: str | None = None,
        limit: int = 10,
        higher_is_better: bool = True,
    ) -> Sequence[Experiment]:
        """Get best experiments by a specific metric."""
        stmt = select(Experiment).where(
            Experiment.task_type == task_type,
            Experiment.primary_metric_name == metric_name,
            Experiment.primary_metric_value.isnot(None),
        )

        if user_id:
            stmt = stmt.where(Experiment.user_id == user_id)

        if higher_is_better:
            stmt = stmt.order_by(desc(Experiment.primary_metric_value))
        else:
            stmt = stmt.order_by(Experiment.primary_metric_value)

        stmt = stmt.limit(limit)
        return self.db.scalars(stmt).all()


class ModelRepository:
    """Repository for Model operations."""

    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, model_id: str) -> Model | None:
        """Get model by ID."""
        return self.db.get(Model, model_id)

    def list_by_user(
        self,
        user_id: str,
        active_only: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[Model]:
        """List models for a user."""
        stmt = select(Model).where(Model.user_id == user_id)

        if active_only:
            stmt = stmt.where(Model.is_active.is_(True))

        stmt = stmt.order_by(desc(Model.created_at)).limit(limit).offset(offset)
        return self.db.scalars(stmt).all()

    def create(
        self,
        name: str,
        estimator_name: str,
        task_type: str,
        blob_path: str,
        user_id: str | None = None,
        description: str | None = None,
        storage_backend: str = "azure_blob",
        file_size_bytes: int | None = None,
    ) -> Model:
        """Create a new model record."""
        model = Model(
            name=name,
            estimator_name=estimator_name,
            task_type=task_type,
            blob_path=blob_path,
            user_id=user_id,
            description=description,
            storage_backend=storage_backend,
            file_size_bytes=file_size_bytes,
        )
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return model

    def deactivate(self, model_id: str) -> bool:
        """Soft-delete a model by deactivating it."""
        model = self.get_by_id(model_id)
        if model:
            model.is_active = False
            self.db.commit()
            return True
        return False

    def link_to_experiment(self, model_id: str, experiment_id: str) -> bool:
        """Link a model to an experiment."""
        model = self.get_by_id(model_id)
        if model:
            # Update the experiment's model_id
            stmt = select(Experiment).where(Experiment.id == experiment_id)
            experiment = self.db.scalar(stmt)
            if experiment:
                experiment.model_id = model_id
                self.db.commit()
                return True
        return False
