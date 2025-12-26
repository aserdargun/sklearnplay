"""CSV upload and data ingestion utilities.

Handles file upload, target selection, dtype detection and user overrides.
"""

from typing import Literal
import pandas as pd
import numpy as np
from io import StringIO

from skplay.core.datasets import DatasetResult, DatasetCard, FeatureInfo, TaskType


def detect_dtype(series: pd.Series) -> Literal["numeric", "categorical", "binary"]:
    """Detect the dtype of a pandas Series."""
    if pd.api.types.is_numeric_dtype(series):
        unique_values = series.dropna().unique()
        if len(unique_values) <= 2 and set(unique_values).issubset({0, 1, True, False}):
            return "binary"
        return "numeric"
    else:
        unique_values = series.dropna().unique()
        if len(unique_values) == 2:
            return "binary"
        return "categorical"


def infer_task_type(y: pd.Series | None, n_classes: int | None = None) -> TaskType:
    """Infer the task type from the target variable."""
    if y is None:
        return "clustering"

    if pd.api.types.is_numeric_dtype(y):
        unique_values = y.dropna().unique()
        # If few unique values, likely classification
        if len(unique_values) <= 10 and len(unique_values) / len(y) < 0.05:
            return "classification"
        return "regression"
    else:
        return "classification"


def parse_csv(
    file_content: str | bytes,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Parse CSV content into a DataFrame."""
    if isinstance(file_content, bytes):
        file_content = file_content.decode(encoding)

    return pd.read_csv(StringIO(file_content), sep=sep)


def create_dataset_from_upload(
    df: pd.DataFrame,
    target_column: str | None = None,
    task_type: TaskType | None = None,
    dataset_name: str = "uploaded",
    description: str = "User-uploaded dataset",
    dtype_overrides: dict[str, Literal["numeric", "categorical", "binary"]] | None = None,
) -> DatasetResult:
    """Create a DatasetResult from an uploaded DataFrame.

    Args:
        df: The uploaded DataFrame
        target_column: Name of the target column (None for unsupervised)
        task_type: Override for task type detection
        dataset_name: Name for the dataset
        description: Description for the dataset
        dtype_overrides: Manual dtype overrides for columns

    Returns:
        DatasetResult with X, y (optional), and card metadata
    """
    dtype_overrides = dtype_overrides or {}

    # Split features and target
    if target_column and target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df.copy()
        y = None
        target_column = None

    # Detect feature types
    features = []
    for col in X.columns:
        if col in dtype_overrides:
            dtype = dtype_overrides[col]
        else:
            dtype = detect_dtype(X[col])

        # Get categories for categorical columns
        categories = None
        if dtype == "categorical":
            categories = X[col].dropna().unique().tolist()

        features.append(FeatureInfo(
            name=col,
            dtype=dtype,
            description=f"Column: {col}",
            categories=categories,
        ))

    # Infer task type if not provided
    if task_type is None:
        task_type = infer_task_type(y)

    # Determine domain
    domain = "general"

    # Create card
    card = DatasetCard(
        name=dataset_name,
        description=description,
        task_type=task_type,
        domain=domain,
        n_samples=len(X),
        n_features=len(X.columns),
        target_name=target_column,
        features=features,
        source="user_upload",
        difficulty="medium",
        tags=["uploaded"],
    )

    return DatasetResult(X=X, y=y, card=card)


def get_column_summary(df: pd.DataFrame) -> dict:
    """Get a summary of DataFrame columns for UI display."""
    summary = {}
    for col in df.columns:
        col_info = {
            "dtype": str(df[col].dtype),
            "inferred_type": detect_dtype(df[col]),
            "n_unique": df[col].nunique(),
            "n_missing": df[col].isna().sum(),
            "missing_pct": round(df[col].isna().mean() * 100, 1),
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["min"] = df[col].min()
            col_info["max"] = df[col].max()
            col_info["mean"] = df[col].mean()
        else:
            col_info["sample_values"] = df[col].dropna().unique()[:5].tolist()

        summary[col] = col_info

    return summary


def validate_upload(df: pd.DataFrame) -> list[str]:
    """Validate an uploaded DataFrame and return any warnings."""
    warnings = []

    if len(df) == 0:
        warnings.append("Dataset is empty")

    if len(df.columns) == 0:
        warnings.append("No columns found")

    # Check for high missing values
    for col in df.columns:
        missing_pct = df[col].isna().mean()
        if missing_pct > 0.5:
            warnings.append(f"Column '{col}' has {missing_pct*100:.0f}% missing values")

    # Check for constant columns
    for col in df.columns:
        if df[col].nunique() <= 1:
            warnings.append(f"Column '{col}' has only one unique value")

    # Check for very high cardinality categorical columns
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() > 100:
                warnings.append(
                    f"Column '{col}' has {df[col].nunique()} unique values - consider encoding or dropping"
                )

    return warnings
