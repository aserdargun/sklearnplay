"""Tests for preprocessing utilities."""

import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

from skplay.core.preprocessing import (
    PreprocessingBuilder,
    identify_column_types,
    encode_target,
    get_preprocessing_options,
)
from skplay.core.datasets import get_dataset


class TestPreprocessingBuilder:
    """Tests for PreprocessingBuilder."""

    def test_build_numeric_pipeline(self):
        """Test building numeric preprocessing pipeline."""
        builder = PreprocessingBuilder(
            numeric_imputer="median",
            numeric_scaler="standard",
        )

        pipeline = builder.build_numeric_pipeline()
        assert pipeline is not None
        assert len(pipeline.steps) == 2

    def test_build_categorical_pipeline(self):
        """Test building categorical preprocessing pipeline."""
        builder = PreprocessingBuilder(
            categorical_imputer="most_frequent",
            categorical_encoder="onehot",
        )

        pipeline = builder.build_categorical_pipeline()
        assert pipeline is not None
        assert len(pipeline.steps) == 2

    def test_build_column_transformer(self):
        """Test building ColumnTransformer."""
        builder = PreprocessingBuilder()

        ct = builder.build_column_transformer(
            numeric_cols=["a", "b"],
            categorical_cols=["c"],
        )

        assert ct is not None
        assert len(ct.transformers) == 2

    def test_build_full_pipeline(self):
        """Test building full preprocessing pipeline."""
        # Load a dataset with mixed types
        data = get_dataset("retail_demand")
        X = data.X

        builder = PreprocessingBuilder(task_type="regression")
        pipeline, numeric_cols, categorical_cols = builder.build_full_pipeline(X)

        assert pipeline is not None
        assert len(numeric_cols) > 0
        assert len(categorical_cols) > 0

        # Test that it can transform
        X_transformed = pipeline.fit_transform(X)
        assert X_transformed is not None
        assert len(X_transformed) == len(X)

    def test_no_scaler(self):
        """Test pipeline with no scaler."""
        builder = PreprocessingBuilder(
            numeric_imputer="median",
            numeric_scaler="none",
        )

        pipeline = builder.build_numeric_pipeline()
        assert len(pipeline.steps) == 1


class TestIdentifyColumnTypes:
    """Tests for column type identification."""

    def test_identify_numeric(self):
        """Test identifying numeric columns."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [1.0, 2.0, 3.0],
            "c": ["x", "y", "z"],
        })

        numeric, categorical = identify_column_types(df)
        assert "a" in numeric
        assert "b" in numeric
        assert "c" in categorical

    def test_force_types(self):
        """Test forcing column types."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [1.0, 2.0, 3.0],
        })

        numeric, categorical = identify_column_types(
            df,
            force_categorical=["a"],
        )

        assert "a" in categorical
        assert "b" in numeric


class TestEncodeTarget:
    """Tests for target encoding."""

    def test_encode_classification_target(self):
        """Test encoding classification target."""
        y = pd.Series(["a", "b", "a", "c"])
        y_encoded, encoder = encode_target(y, "classification")

        assert encoder is not None
        assert len(np.unique(y_encoded)) == 3
        assert all(isinstance(v, (int, np.integer)) for v in y_encoded)

    def test_encode_regression_target(self):
        """Test that regression target is not encoded."""
        y = pd.Series([1.0, 2.0, 3.0])
        y_encoded, encoder = encode_target(y, "regression")

        assert encoder is None
        assert np.allclose(y_encoded, y.values)

    def test_encode_numeric_classification(self):
        """Test numeric classification target."""
        y = pd.Series([0, 1, 0, 1])
        y_encoded, encoder = encode_target(y, "classification")

        assert encoder is None  # Already numeric
        assert np.array_equal(y_encoded, y.values)


class TestPreprocessingOptions:
    """Tests for preprocessing options by level."""

    def test_beginner_options(self):
        """Test beginner level options."""
        options = get_preprocessing_options("beginner")

        assert "none" in options["numeric_imputers"]
        assert len(options["feature_selectors"]) == 1  # Only "none"

    def test_advanced_options(self):
        """Test advanced level has more options."""
        beginner = get_preprocessing_options("beginner")
        advanced = get_preprocessing_options("advanced")

        assert len(advanced["numeric_imputers"]) > len(beginner["numeric_imputers"])
        assert len(advanced["feature_selectors"]) > len(beginner["feature_selectors"])
