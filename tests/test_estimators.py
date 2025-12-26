"""Tests for estimator registry."""

import pytest
from sklearn.base import BaseEstimator

from skplay.core.estimators import (
    EstimatorRegistry,
    get_estimators_for_task,
    get_estimator_names,
    get_estimator_class,
    create_estimator,
    get_default_estimator,
)


class TestEstimatorRegistry:
    """Tests for EstimatorRegistry."""

    def test_get_classification_estimators(self):
        """Test getting classification estimators."""
        estimators = get_estimators_for_task("classification")
        assert len(estimators) > 0

        names = [e.name for e in estimators]
        assert "Logistic Regression" in names
        assert "Random Forest" in names

    def test_get_regression_estimators(self):
        """Test getting regression estimators."""
        estimators = get_estimators_for_task("regression")
        assert len(estimators) > 0

        names = [e.name for e in estimators]
        assert "Linear Regression" in names
        assert "Random Forest Regressor" in names

    def test_get_clustering_estimators(self):
        """Test getting clustering estimators."""
        estimators = get_estimators_for_task("clustering")
        assert len(estimators) > 0

        names = [e.name for e in estimators]
        assert "K-Means" in names
        assert "DBSCAN" in names

    def test_get_outlier_estimators(self):
        """Test getting outlier detection estimators."""
        estimators = get_estimators_for_task("outlier_detection")
        assert len(estimators) > 0

        names = [e.name for e in estimators]
        assert "Isolation Forest" in names

    def test_level_filtering(self):
        """Test that level filtering works."""
        beginner = get_estimators_for_task("classification", "beginner")
        advanced = get_estimators_for_task("classification", "advanced")

        assert len(advanced) >= len(beginner)

        # Beginner should have basic estimators
        beginner_names = [e.name for e in beginner]
        assert "Logistic Regression" in beginner_names
        assert "Random Forest" in beginner_names

    def test_get_estimator_names(self):
        """Test getting estimator names."""
        names = get_estimator_names("classification")
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_get_estimator_class(self):
        """Test getting estimator class."""
        cls = get_estimator_class("classification", "Logistic Regression")
        assert cls is not None

        # Create instance
        instance = cls()
        assert isinstance(instance, BaseEstimator)

    def test_create_estimator(self):
        """Test creating estimator with parameters."""
        estimator = create_estimator(
            "classification",
            "Random Forest",
            n_estimators=50,
            random_state=42,
        )

        assert estimator is not None
        assert estimator.n_estimators == 50
        assert estimator.random_state == 42

    def test_create_estimator_invalid(self):
        """Test creating invalid estimator raises error."""
        with pytest.raises(ValueError):
            create_estimator("classification", "NonexistentModel")

    def test_default_estimator(self):
        """Test getting default estimator."""
        default_clf = get_default_estimator("classification")
        assert default_clf == "Random Forest"

        default_reg = get_default_estimator("regression")
        assert default_reg == "Random Forest Regressor"

        default_cluster = get_default_estimator("clustering")
        assert default_cluster == "K-Means"

    def test_estimator_info_fields(self):
        """Test estimator info has required fields."""
        estimators = get_estimators_for_task("classification")

        for info in estimators:
            assert info.name
            assert info.class_ is not None
            assert info.description
            assert info.level in ("beginner", "intermediate", "advanced")
            assert isinstance(info.key_params, list)
