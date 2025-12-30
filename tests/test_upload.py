"""Tests for CSV upload and data ingestion utilities."""

import numpy as np
import pandas as pd

from skplay.core.upload import (
    create_dataset_from_upload,
    detect_dtype,
    get_column_summary,
    infer_task_type,
    parse_csv,
    validate_upload,
)


class TestDetectDtype:
    """Tests for detect_dtype function."""

    def test_numeric_column(self):
        """Test detecting numeric column."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert detect_dtype(series) == "numeric"

    def test_integer_column(self):
        """Test detecting integer column."""
        series = pd.Series([1, 2, 3, 4, 5])
        assert detect_dtype(series) == "numeric"

    def test_binary_numeric(self):
        """Test detecting binary numeric column."""
        series = pd.Series([0, 1, 0, 1, 0, 1])
        assert detect_dtype(series) == "binary"

    def test_categorical_column(self):
        """Test detecting categorical column."""
        series = pd.Series(["a", "b", "c", "a", "b", "c", "d"])
        assert detect_dtype(series) == "categorical"

    def test_binary_categorical(self):
        """Test detecting binary categorical column."""
        series = pd.Series(["yes", "no", "yes", "no", "yes"])
        assert detect_dtype(series) == "binary"

    def test_with_missing_values(self):
        """Test dtype detection with missing values."""
        series = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        assert detect_dtype(series) == "numeric"


class TestInferTaskType:
    """Tests for infer_task_type function."""

    def test_classification_string_target(self):
        """Test inferring classification from string target."""
        y = pd.Series(["cat", "dog", "cat", "dog", "cat"])
        assert infer_task_type(y) == "classification"

    def test_classification_few_unique_numeric(self):
        """Test inferring classification from numeric with few unique values."""
        # Need enough samples to pass the ratio threshold (len(unique) / len(y) < 0.05)
        # 3 unique values with 100 samples: 3/100 = 0.03 < 0.05
        y = pd.Series([0, 1, 2] * 34)  # 102 samples, 3 unique values
        assert infer_task_type(y) == "classification"

    def test_regression_continuous(self):
        """Test inferring regression from continuous target."""
        y = pd.Series(np.random.randn(100))
        assert infer_task_type(y) == "regression"

    def test_clustering_none_target(self):
        """Test inferring clustering when no target."""
        assert infer_task_type(None) == "clustering"


class TestParseCsv:
    """Tests for parse_csv function."""

    def test_parse_string_csv(self):
        """Test parsing CSV from string."""
        csv_content = "a,b,c\n1,2,3\n4,5,6\n7,8,9"
        df = parse_csv(csv_content)

        assert list(df.columns) == ["a", "b", "c"]
        assert len(df) == 3

    def test_parse_bytes_csv(self):
        """Test parsing CSV from bytes."""
        csv_content = b"a,b,c\n1,2,3\n4,5,6"
        df = parse_csv(csv_content)

        assert list(df.columns) == ["a", "b", "c"]
        assert len(df) == 2

    def test_custom_separator(self):
        """Test parsing with custom separator."""
        csv_content = "a;b;c\n1;2;3\n4;5;6"
        df = parse_csv(csv_content, sep=";")

        assert list(df.columns) == ["a", "b", "c"]


class TestCreateDatasetFromUpload:
    """Tests for create_dataset_from_upload function."""

    def test_with_target(self):
        """Test creating dataset with target column."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [5, 4, 3, 2, 1],
            "target": [0, 1, 0, 1, 0]
        })

        result = create_dataset_from_upload(df, target_column="target")

        assert result.X.shape == (5, 2)
        assert result.y is not None
        assert len(result.y) == 5
        assert result.card.target_name == "target"

    def test_without_target(self):
        """Test creating dataset without target (for clustering)."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [5, 4, 3, 2, 1]
        })

        result = create_dataset_from_upload(df, target_column=None)

        assert result.X.shape == (5, 2)
        assert result.y is None
        assert result.card.task_type == "clustering"

    def test_with_task_type_override(self):
        """Test overriding inferred task type."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 0]
        })

        result = create_dataset_from_upload(
            df,
            target_column="target",
            task_type="classification"
        )

        assert result.card.task_type == "classification"

    def test_with_dtype_overrides(self):
        """Test overriding column types."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": ["a", "b", "c", "a", "b"],
            "target": [0, 1, 0, 1, 0]
        })

        result = create_dataset_from_upload(
            df,
            target_column="target",
            dtype_overrides={"feature1": "categorical"}
        )

        # Find feature1 in features
        feature1_info = next(f for f in result.card.features if f.name == "feature1")
        assert feature1_info.dtype == "categorical"

    def test_dataset_card_metadata(self):
        """Test dataset card has correct metadata."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 0]
        })

        result = create_dataset_from_upload(
            df,
            target_column="target",
            dataset_name="my_dataset",
            description="My test dataset"
        )

        assert result.card.name == "my_dataset"
        assert result.card.description == "My test dataset"
        assert result.card.n_samples == 5
        assert result.card.n_features == 1
        assert result.card.source == "user_upload"


class TestGetColumnSummary:
    """Tests for get_column_summary function."""

    def test_numeric_column_summary(self):
        """Test summary for numeric column."""
        df = pd.DataFrame({
            "numeric": [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        summary = get_column_summary(df)

        assert "numeric" in summary
        assert "min" in summary["numeric"]
        assert "max" in summary["numeric"]
        assert "mean" in summary["numeric"]
        assert summary["numeric"]["min"] == 1.0
        assert summary["numeric"]["max"] == 5.0

    def test_categorical_column_summary(self):
        """Test summary for categorical column."""
        df = pd.DataFrame({
            "category": ["a", "b", "c", "a", "b"]
        })

        summary = get_column_summary(df)

        assert "category" in summary
        assert "sample_values" in summary["category"]
        assert "n_unique" in summary["category"]

    def test_missing_values_counted(self):
        """Test that missing values are counted."""
        df = pd.DataFrame({
            "feature": [1.0, 2.0, np.nan, 4.0, np.nan]
        })

        summary = get_column_summary(df)

        assert summary["feature"]["n_missing"] == 2
        assert summary["feature"]["missing_pct"] == 40.0


class TestValidateUpload:
    """Tests for validate_upload function."""

    def test_valid_dataframe(self):
        """Test validation of valid DataFrame."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1]
        })

        warnings = validate_upload(df)
        assert len(warnings) == 0

    def test_empty_dataframe(self):
        """Test validation of empty DataFrame."""
        df = pd.DataFrame()

        warnings = validate_upload(df)
        assert any("empty" in w.lower() for w in warnings)

    def test_high_missing_values(self):
        """Test warning for high missing values."""
        df = pd.DataFrame({
            "feature": [1.0, np.nan, np.nan, np.nan, np.nan, np.nan]
        })

        warnings = validate_upload(df)
        assert any("missing" in w.lower() for w in warnings)

    def test_constant_column(self):
        """Test warning for constant column."""
        df = pd.DataFrame({
            "constant": [1, 1, 1, 1, 1],
            "normal": [1, 2, 3, 4, 5]
        })

        warnings = validate_upload(df)
        assert any("unique" in w.lower() for w in warnings)

    def test_high_cardinality_categorical(self):
        """Test warning for high cardinality categorical."""
        df = pd.DataFrame({
            "high_card": [f"val_{i}" for i in range(200)]
        })

        warnings = validate_upload(df)
        assert any("unique values" in w.lower() for w in warnings)

    def test_multiple_issues(self):
        """Test detection of multiple issues."""
        df = pd.DataFrame({
            "constant": [1, 1, 1, 1, 1],
            "high_missing": [1.0, np.nan, np.nan, np.nan, np.nan]
        })

        warnings = validate_upload(df)
        assert len(warnings) >= 2
