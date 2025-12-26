"""Tests for the dataset registry."""

import pytest
import pandas as pd
import numpy as np

from skplay.core.datasets import (
    DatasetRegistry,
    get_dataset,
    list_datasets,
    list_datasets_by_domain,
    list_datasets_by_task,
)


class TestDatasetRegistry:
    """Tests for DatasetRegistry."""

    def test_list_all_datasets(self):
        """Test that datasets are registered."""
        datasets = list_datasets()
        assert len(datasets) > 0
        assert "iris" in datasets
        assert "breast_cancer" in datasets

    def test_list_by_domain(self):
        """Test filtering by domain."""
        power_datasets = list_datasets_by_domain("power")
        assert len(power_datasets) > 0
        assert "power_failure" in power_datasets

        retail_datasets = list_datasets_by_domain("retail")
        assert len(retail_datasets) > 0

    def test_list_by_task(self):
        """Test filtering by task type."""
        classification = list_datasets_by_task("classification")
        assert len(classification) > 0
        assert "iris" in classification

        regression = list_datasets_by_task("regression")
        assert len(regression) > 0
        assert "diabetes" in regression

        clustering = list_datasets_by_task("clustering")
        assert len(clustering) > 0

    def test_get_dataset_iris(self):
        """Test loading iris dataset."""
        result = get_dataset("iris")

        assert result.X is not None
        assert result.y is not None
        assert isinstance(result.X, pd.DataFrame)
        assert isinstance(result.y, pd.Series)
        assert len(result.X) == len(result.y)
        assert result.card.task_type == "classification"

    def test_get_dataset_diabetes(self):
        """Test loading diabetes dataset."""
        result = get_dataset("diabetes")

        assert result.X is not None
        assert result.y is not None
        assert result.card.task_type == "regression"

    def test_get_dataset_clustering(self):
        """Test loading clustering dataset."""
        result = get_dataset("retail_segmentation")

        assert result.X is not None
        assert result.y is None  # Clustering has no target
        assert result.card.task_type == "clustering"

    def test_get_dataset_card(self):
        """Test dataset card metadata."""
        result = get_dataset("iris")
        card = result.card

        assert card.name == "iris"
        assert card.n_samples > 0
        assert card.n_features > 0
        assert len(card.features) == card.n_features
        assert card.domain in ["general", "power", "retail", "finance", "healthcare"]

    def test_invalid_dataset(self):
        """Test that invalid dataset raises error."""
        with pytest.raises(ValueError):
            get_dataset("nonexistent_dataset")

    def test_synthetic_datasets(self):
        """Test synthetic datasets are well-formed."""
        synthetic_datasets = [
            "power_failure",
            "power_efficiency",
            "retail_churn",
            "retail_demand",
            "credit_risk",
            "fraud_detection",
        ]

        for name in synthetic_datasets:
            result = get_dataset(name)
            assert result.X is not None
            assert not result.X.isnull().any().any(), f"{name} has NaN values"
            assert len(result.X) > 0
