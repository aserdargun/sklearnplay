"""Preprocessing utilities using sklearn idioms.

Provides ColumnTransformer builders and preprocessing pipeline assembly.
"""

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from skplay.core.datasets import TaskType

# Available preprocessing options
NUMERIC_SCALERS = {
    "none": None,
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "maxabs": MaxAbsScaler,
}

NUMERIC_IMPUTERS = {
    "none": None,
    "mean": lambda: SimpleImputer(strategy="mean"),
    "median": lambda: SimpleImputer(strategy="median"),
    "most_frequent": lambda: SimpleImputer(strategy="most_frequent"),
    "constant_zero": lambda: SimpleImputer(strategy="constant", fill_value=0),
    "knn": lambda: KNNImputer(n_neighbors=5),
}

CATEGORICAL_ENCODERS = {
    "onehot": lambda: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "ordinal": lambda: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
}

CATEGORICAL_IMPUTERS = {
    "none": None,
    "most_frequent": lambda: SimpleImputer(strategy="most_frequent"),
    "constant_missing": lambda: SimpleImputer(strategy="constant", fill_value="missing"),
}

FEATURE_SELECTORS = {
    "none": None,
    "selectkbest_f": "selectkbest_f",
    "selectkbest_mutual_info": "selectkbest_mutual_info",
    "selectpercentile_f": "selectpercentile_f",
}

DIMENSIONALITY_REDUCERS = {
    "none": None,
    "pca": PCA,
    "truncated_svd": TruncatedSVD,
}


def identify_column_types(
    X: pd.DataFrame,
    force_categorical: list[str] | None = None,
    force_numeric: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Identify numeric and categorical columns.

    Args:
        X: Input DataFrame
        force_categorical: Column names to treat as categorical
        force_numeric: Column names to treat as numeric

    Returns:
        Tuple of (numeric_columns, categorical_columns)
    """
    force_categorical = force_categorical or []
    force_numeric = force_numeric or []

    numeric_cols = []
    categorical_cols = []

    for col in X.columns:
        # Skip only None or empty string column names, NOT falsy values like integer 0
        if col is None or (isinstance(col, str) and not col.strip()):
            continue
        if col in force_categorical:
            categorical_cols.append(col)
        elif col in force_numeric:
            numeric_cols.append(col)
        elif pd.api.types.is_numeric_dtype(X[col]):
            numeric_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(X[col]):
            # Datetime columns can't be used directly by sklearn transformers
            # Skip them - they need special handling (conversion to numeric features)
            continue
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


class PreprocessingBuilder:
    """Builder for creating preprocessing pipelines.

    Uses sklearn idioms: Pipeline, ColumnTransformer.
    """

    def __init__(
        self,
        numeric_imputer: str = "median",
        numeric_scaler: str = "standard",
        categorical_imputer: str = "most_frequent",
        categorical_encoder: str = "onehot",
        feature_selector: str = "none",
        feature_selector_k: int = 10,
        feature_selector_percentile: int = 50,
        dim_reducer: str = "none",
        dim_reducer_n_components: int | float = 0.95,
        task_type: TaskType = "classification",
    ):
        """Initialize preprocessing builder.

        Args:
            numeric_imputer: Imputation strategy for numeric columns
            numeric_scaler: Scaling strategy for numeric columns
            categorical_imputer: Imputation strategy for categorical columns
            categorical_encoder: Encoding strategy for categorical columns
            feature_selector: Feature selection method
            feature_selector_k: Number of features to select (for SelectKBest)
            feature_selector_percentile: Percentile of features to keep
            dim_reducer: Dimensionality reduction method
            dim_reducer_n_components: Number of components for dim reduction
            task_type: Task type (affects feature selection scoring)
        """
        self.numeric_imputer = numeric_imputer
        self.numeric_scaler = numeric_scaler
        self.categorical_imputer = categorical_imputer
        self.categorical_encoder = categorical_encoder
        self.feature_selector = feature_selector
        self.feature_selector_k = feature_selector_k
        self.feature_selector_percentile = feature_selector_percentile
        self.dim_reducer = dim_reducer
        self.dim_reducer_n_components = dim_reducer_n_components
        self.task_type = task_type

    def build_numeric_pipeline(self) -> Pipeline | None:
        """Build preprocessing pipeline for numeric columns."""
        steps = []

        # Imputation
        if self.numeric_imputer != "none":
            imputer_fn = NUMERIC_IMPUTERS[self.numeric_imputer]
            if imputer_fn is not None:
                steps.append(("imputer", imputer_fn()))

        # Scaling
        if self.numeric_scaler != "none":
            scaler = NUMERIC_SCALERS[self.numeric_scaler]()
            steps.append(("scaler", scaler))

        if not steps:
            return None

        return Pipeline(steps)

    def build_categorical_pipeline(self) -> Pipeline | None:
        """Build preprocessing pipeline for categorical columns."""
        steps = []

        # Imputation
        if self.categorical_imputer != "none":
            imputer_fn = CATEGORICAL_IMPUTERS[self.categorical_imputer]
            if imputer_fn is not None:
                steps.append(("imputer", imputer_fn()))

        # Encoding
        encoder = CATEGORICAL_ENCODERS[self.categorical_encoder]()
        steps.append(("encoder", encoder))

        return Pipeline(steps)

    def build_column_transformer(
        self,
        numeric_cols: list[str],
        categorical_cols: list[str],
    ) -> ColumnTransformer:
        """Build a ColumnTransformer for mixed-type data.

        Args:
            numeric_cols: List of numeric column names
            categorical_cols: List of categorical column names

        Returns:
            ColumnTransformer that handles both types
        """
        transformers = []

        # Filter only None or empty strings, preserve falsy values like integer 0
        numeric_cols = [c for c in numeric_cols if c is not None and c != '']
        categorical_cols = [c for c in categorical_cols if c is not None and c != '']

        if numeric_cols:
            num_pipeline = self.build_numeric_pipeline()
            if num_pipeline:
                transformers.append(("numeric", num_pipeline, numeric_cols))
            else:
                transformers.append(("numeric", "passthrough", numeric_cols))

        if categorical_cols:
            cat_pipeline = self.build_categorical_pipeline()
            if cat_pipeline:
                transformers.append(("categorical", cat_pipeline, categorical_cols))

        # If no transformers, use passthrough to avoid empty output
        if not transformers:
            return ColumnTransformer(
                transformers=[("passthrough", "passthrough", slice(None))],
                remainder="drop",
                verbose_feature_names_out=False,
            )

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def build_feature_selector(self) -> object | None:
        """Build feature selector if configured."""
        if self.feature_selector == "none":
            return None

        if self.task_type in ("classification", "outlier_detection"):
            f_score_func = f_classif
            mi_score_func = mutual_info_classif
        else:
            f_score_func = f_regression
            mi_score_func = mutual_info_regression

        if self.feature_selector == "selectkbest_f":
            return SelectKBest(score_func=f_score_func, k=self.feature_selector_k)
        elif self.feature_selector == "selectkbest_mutual_info":
            return SelectKBest(score_func=mi_score_func, k=self.feature_selector_k)
        elif self.feature_selector == "selectpercentile_f":
            return SelectPercentile(
                score_func=f_score_func, percentile=self.feature_selector_percentile
            )

        return None

    def build_dim_reducer(self) -> object | None:
        """Build dimensionality reducer if configured."""
        if self.dim_reducer == "none":
            return None

        reducer_class = DIMENSIONALITY_REDUCERS[self.dim_reducer]
        return reducer_class(n_components=self.dim_reducer_n_components)

    def build_full_pipeline(
        self,
        X: pd.DataFrame,
        force_categorical: list[str] | None = None,
        force_numeric: list[str] | None = None,
    ) -> tuple[Pipeline, list[str], list[str]]:
        """Build complete preprocessing pipeline.

        Args:
            X: Input DataFrame (used to identify column types)
            force_categorical: Column names to force as categorical
            force_numeric: Column names to force as numeric

        Returns:
            Tuple of (Pipeline, numeric_cols, categorical_cols)
        """
        numeric_cols, categorical_cols = identify_column_types(X, force_categorical, force_numeric)

        # Validate that columns actually exist in the DataFrame
        existing_cols = set(X.columns)
        numeric_cols = [c for c in numeric_cols if c in existing_cols]
        categorical_cols = [c for c in categorical_cols if c in existing_cols]

        steps = []

        # Column transformer for mixed types
        col_transformer = self.build_column_transformer(numeric_cols, categorical_cols)
        steps.append(("preprocessor", col_transformer))

        # Feature selection (only for supervised tasks)
        if self.task_type not in ("clustering",):
            feature_selector = self.build_feature_selector()
            if feature_selector:
                steps.append(("feature_selection", feature_selector))

        # Dimensionality reduction
        dim_reducer = self.build_dim_reducer()
        if dim_reducer:
            steps.append(("dim_reduction", dim_reducer))

        pipeline = Pipeline(steps)

        return pipeline, numeric_cols, categorical_cols


def encode_target(y: pd.Series, task_type: TaskType) -> tuple[np.ndarray, LabelEncoder | None]:
    """Encode target variable if needed.

    Args:
        y: Target series
        task_type: The task type

    Returns:
        Tuple of (encoded_y, encoder_or_None)
    """
    if task_type == "regression":
        return y.values, None

    if pd.api.types.is_numeric_dtype(y):
        return y.values, None

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder


def get_preprocessing_options(level: Literal["beginner", "intermediate", "advanced"]) -> dict:
    """Get available preprocessing options based on user level.

    Args:
        level: User expertise level

    Returns:
        Dictionary of available options per category
    """
    if level == "beginner":
        return {
            "numeric_imputers": ["none", "mean", "median"],
            "numeric_scalers": ["none", "standard", "minmax"],
            "categorical_imputers": ["none", "most_frequent"],
            "categorical_encoders": ["onehot"],
            "feature_selectors": ["none"],
            "dim_reducers": ["none"],
        }
    elif level == "intermediate":
        return {
            "numeric_imputers": ["none", "mean", "median", "most_frequent", "knn"],
            "numeric_scalers": ["none", "standard", "minmax", "robust"],
            "categorical_imputers": ["none", "most_frequent", "constant_missing"],
            "categorical_encoders": ["onehot", "ordinal"],
            "feature_selectors": ["none", "selectkbest_f"],
            "dim_reducers": ["none", "pca"],
        }
    else:  # advanced
        return {
            "numeric_imputers": list(NUMERIC_IMPUTERS.keys()),
            "numeric_scalers": list(NUMERIC_SCALERS.keys()),
            "categorical_imputers": list(CATEGORICAL_IMPUTERS.keys()),
            "categorical_encoders": list(CATEGORICAL_ENCODERS.keys()),
            "feature_selectors": list(FEATURE_SELECTORS.keys()),
            "dim_reducers": list(DIMENSIONALITY_REDUCERS.keys()),
        }


def get_preprocessing_help() -> dict:
    """Get help text for preprocessing options."""
    return {
        "numeric_imputers": {
            "none": "No imputation - fails if missing values present",
            "mean": "Replace missing with column mean",
            "median": "Replace missing with column median (robust to outliers)",
            "most_frequent": "Replace missing with most common value",
            "constant_zero": "Replace missing with zero",
            "knn": "Use K-nearest neighbors to impute missing values",
        },
        "numeric_scalers": {
            "none": "No scaling - keep original values",
            "standard": "Zero mean, unit variance (z-score normalization)",
            "minmax": "Scale to [0, 1] range",
            "robust": "Scale using median and IQR (robust to outliers)",
            "maxabs": "Scale by maximum absolute value",
        },
        "categorical_encoders": {
            "onehot": "Create binary columns for each category",
            "ordinal": "Encode as integers (assumes ordering)",
        },
        "feature_selectors": {
            "none": "Keep all features",
            "selectkbest_f": "Select K best features using F-test",
            "selectkbest_mutual_info": "Select K best using mutual information",
            "selectpercentile_f": "Select top percentile using F-test",
        },
        "dim_reducers": {
            "none": "No dimensionality reduction",
            "pca": "Principal Component Analysis",
            "truncated_svd": "Truncated SVD (works with sparse data)",
        },
    }
