"""SQLAlchemy ORM models for sklearn-playground."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class User(Base):
    """User model for Azure Entra ID authenticated users."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    azure_oid: Mapped[str] = mapped_column(String(36), unique=True, index=True, nullable=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(255))
    preferred_level: Mapped[str] = mapped_column(String(20), default="beginner")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_login: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    experiments: Mapped[list["Experiment"]] = relationship(
        "Experiment", back_populates="user", cascade="all, delete-orphan"
    )
    models: Mapped[list["Model"]] = relationship(
        "Model", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email})>"


class Experiment(Base):
    """Experiment tracking model for ML runs."""

    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=True)
    session_id: Mapped[str] = mapped_column(String(36), index=True, nullable=True)

    # Experiment metadata
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    task_type: Mapped[str] = mapped_column(String(50))  # classification, regression, clustering
    dataset_name: Mapped[str] = mapped_column(String(255))

    # Model configuration
    estimator_name: Mapped[str] = mapped_column(String(100))
    estimator_params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    preprocessing_config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    # Results
    metrics: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)
    primary_metric_name: Mapped[str | None] = mapped_column(String(50), nullable=True)
    primary_metric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    training_time_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Artifacts
    model_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("models.id"), nullable=True
    )
    code_snippet: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Status and timestamps
    status: Mapped[str] = mapped_column(String(20), default="completed")  # running, completed, failed
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Tags for filtering
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="experiments")
    model: Mapped["Model"] = relationship("Model", back_populates="experiment", uselist=False)

    __table_args__ = (
        Index("ix_experiments_user_created", "user_id", "created_at"),
        Index("ix_experiments_task_metric", "task_type", "primary_metric_value"),
    )

    def __repr__(self) -> str:
        return f"<Experiment(id={self.id}, name={self.name}, estimator={self.estimator_name})>"


class Model(Base):
    """Trained model storage metadata."""

    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=True)

    # Model metadata
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    estimator_name: Mapped[str] = mapped_column(String(100))
    task_type: Mapped[str] = mapped_column(String(50))

    # Storage location
    storage_backend: Mapped[str] = mapped_column(String(20), default="azure_blob")  # azure_blob, local
    blob_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(nullable=True)

    # Versioning
    version: Mapped[int] = mapped_column(default=1)
    is_active: Mapped[bool] = mapped_column(default=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="models")
    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="model")

    __table_args__ = (Index("ix_models_user_active", "user_id", "is_active"),)

    def __repr__(self) -> str:
        return f"<Model(id={self.id}, name={self.name}, version={self.version})>"
