"""CSV upload and data ingestion utilities.

Handles file upload, target selection, dtype detection and user overrides.
"""

from io import StringIO
from typing import Literal

import pandas as pd

from skplay.core.datasets import DatasetCard, DatasetResult, Domain, FeatureInfo, TaskType


def detect_datetime_column(df: pd.DataFrame) -> str | None:
    """Detect a datetime column in the DataFrame.

    Looks for columns that are already datetime type or can be parsed as datetime.
    Common datetime column names are prioritized.

    Args:
        df: Input DataFrame

    Returns:
        Column name if a datetime column is found, None otherwise
    """
    # Common datetime column names (case-insensitive)
    datetime_names = {
        "timestamp", "datetime", "date", "time", "created_at", "updated_at",
        "created", "updated", "ts", "dt", "event_time", "event_date",
        "start_time", "end_time", "start_date", "end_date",
    }

    # First, check for columns already parsed as datetime
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

    # Then, look for columns with datetime-like names that can be parsed
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if col_lower in datetime_names:
            try:
                pd.to_datetime(df[col], errors="raise")
                return col
            except (ValueError, TypeError):
                continue

    # Finally, try to detect any column that looks like datetime
    for col in df.columns:
        # Skip numeric columns (could be Unix timestamps, handle separately)
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        # Skip columns with too many unique values relative to string length
        if df[col].dtype == object:
            try:
                # Sample a few values to check if they parse as datetime
                sample = df[col].dropna().head(10)
                if len(sample) > 0:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    # If most values parse successfully, it's likely a datetime column
                    if parsed.notna().mean() > 0.8:
                        return col
            except (ValueError, TypeError):
                continue

    return None


def detect_dtype(series: pd.Series) -> Literal["numeric", "categorical", "binary"]:
    """Detect the dtype of a pandas Series."""
    if pd.api.types.is_numeric_dtype(series):
        unique_values = series.dropna().unique()
        if len(unique_values) <= 2 and set(unique_values).issubset({0, 1}):
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
    datetime_column: str | None = None,
    task_type: TaskType | None = None,
    dataset_name: str = "uploaded",
    description: str = "User-uploaded dataset",
    dtype_overrides: dict[str, Literal["numeric", "categorical", "binary"]] | None = None,
) -> DatasetResult:
    """Create a DatasetResult from an uploaded DataFrame.

    Args:
        df: The uploaded DataFrame
        target_column: Name of the target column (None for unsupervised)
        datetime_column: Name of column to use as datetime index (None to skip)
        task_type: Override for task type detection
        dataset_name: Name for the dataset
        description: Description for the dataset
        dtype_overrides: Manual dtype overrides for columns

    Returns:
        DatasetResult with X, y (optional), and card metadata
    """
    dtype_overrides = dtype_overrides or {}

    # Handle datetime column - parse, sort, then convert to numeric for ML compatibility
    if datetime_column and datetime_column in df.columns:
        df = df.copy()
        df[datetime_column] = pd.to_datetime(df[datetime_column], errors="coerce")
        # Sort by datetime
        df = df.sort_values(datetime_column).reset_index(drop=True)
        # DEBUG: Log datetime parsing results
        nat_count = df[datetime_column].isna().sum()
        print(f"[DEBUG upload.py] Datetime '{datetime_column}': {len(df) - nat_count}/{len(df)} parsed successfully")
        if nat_count == len(df):
            print(f"[DEBUG upload.py] WARNING: All values in '{datetime_column}' failed to parse!")
        # Convert datetime to Unix timestamp (seconds) for ML compatibility
        # sklearn transformers can't handle datetime64 directly
        df[datetime_column] = df[datetime_column].astype("int64") // 10**9  # nanoseconds to seconds
        print(f"[DEBUG upload.py] Converted '{datetime_column}' to Unix timestamp (numeric)")

    # Split features and target
    if target_column and target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df.copy()
        y = None
        target_column = None

    # Validate that we have at least one feature column
    if len(X.columns) == 0:
        raise ValueError(
            "Dataset has no feature columns. Please ensure your CSV has at least one "
            "column besides the target column."
        )

    # DEBUG: Log X shape after creation
    print(f"[DEBUG upload.py] After split - X.shape: {X.shape}, columns: {list(X.columns)}")
    print(f"[DEBUG upload.py] Column dtypes: {dict(X.dtypes)}")

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

        features.append(
            FeatureInfo(
                name=col,
                dtype=dtype,
                description=f"Column: {col}",
                categories=categories,
            )
        )

    # Infer task type if not provided
    if task_type is None:
        task_type = infer_task_type(y)

    # Determine domain
    domain: Domain = "general"

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
            warnings.append(f"Column '{col}' has {missing_pct * 100:.0f}% missing values")

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
