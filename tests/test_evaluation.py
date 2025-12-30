"""Tests for evaluation utilities."""

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

from skplay.core.evaluation import (
    compute_classification_metrics,
    compute_clustering_metrics,
    compute_outlier_metrics,
    compute_regression_metrics,
    evaluate_model,
    get_metrics_for_task,
    plot_actual_vs_predicted,
    plot_cluster_visualization,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_learning_curve,
    plot_outlier_scores,
    plot_precision_recall_curve,
    plot_residuals,
    plot_roc_curve,
    run_cross_validation,
)


class TestGetMetricsForTask:
    """Tests for get_metrics_for_task function."""

    def test_classification_metrics(self):
        """Test metrics for classification task."""
        metrics = get_metrics_for_task("classification")
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    def test_regression_metrics(self):
        """Test metrics for regression task."""
        metrics = get_metrics_for_task("regression")
        assert "r2" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics

    def test_clustering_metrics(self):
        """Test metrics for clustering task."""
        metrics = get_metrics_for_task("clustering")
        assert "silhouette" in metrics
        assert "calinski_harabasz" in metrics

    def test_outlier_detection_metrics(self):
        """Test metrics for outlier detection task."""
        metrics = get_metrics_for_task("outlier_detection")
        assert "precision" in metrics
        assert "recall" in metrics

    def test_invalid_task(self):
        """Test invalid task type returns empty list."""
        metrics = get_metrics_for_task("invalid_task")  # type: ignore
        assert metrics == []


class TestComputeClassificationMetrics:
    """Tests for compute_classification_metrics function."""

    def test_binary_classification(self):
        """Test binary classification metrics."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])

        metrics = compute_classification_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_with_probabilities(self):
        """Test classification metrics with probabilities."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8],
                           [0.7, 0.3], [0.4, 0.6], [0.3, 0.7], [0.1, 0.9]])

        metrics = compute_classification_metrics(y_true, y_pred, y_proba)

        assert "roc_auc" in metrics
        assert "pr_auc" in metrics

    def test_multiclass_classification(self):
        """Test multiclass classification metrics."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2])

        metrics = compute_classification_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1


class TestComputeRegressionMetrics:
    """Tests for compute_regression_metrics function."""

    def test_regression_metrics(self):
        """Test regression metrics calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

        metrics = compute_regression_metrics(y_true, y_pred)

        assert "r2" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "mape" in metrics

    def test_perfect_prediction(self):
        """Test metrics for perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()

        metrics = compute_regression_metrics(y_true, y_pred)

        assert metrics["r2"] == pytest.approx(1.0)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["rmse"] == pytest.approx(0.0)


class TestComputeClusteringMetrics:
    """Tests for compute_clustering_metrics function."""

    def test_clustering_metrics(self):
        """Test clustering metrics calculation."""
        X = np.array([[1, 2], [1, 3], [2, 2], [8, 7], [8, 8], [9, 8]])
        labels = np.array([0, 0, 0, 1, 1, 1])

        metrics = compute_clustering_metrics(X, labels)

        assert "silhouette" in metrics
        assert "n_clusters" in metrics
        assert metrics["n_clusters"] == 2

    def test_single_cluster(self):
        """Test metrics with single cluster returns only n_clusters."""
        X = np.array([[1, 2], [1, 3], [2, 2]])
        labels = np.array([0, 0, 0])

        metrics = compute_clustering_metrics(X, labels)

        # With single cluster, silhouette etc. cannot be computed
        assert "n_clusters" in metrics
        assert metrics["n_clusters"] == 1


class TestComputeOutlierMetrics:
    """Tests for compute_outlier_metrics function."""

    def test_outlier_metrics(self):
        """Test outlier detection metrics."""
        y_true = np.array([0, 0, 0, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 1, 1, -1, 1, 1, -1, 1])  # -1 = outlier

        metrics = compute_outlier_metrics(y_true, y_pred)

        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    def test_with_scores(self):
        """Test outlier metrics with anomaly scores."""
        y_true = np.array([0, 0, 0, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 1, 1, -1, 1, 1, -1, 1])
        scores = np.array([0.1, 0.2, 0.15, 0.9, 0.1, 0.2, 0.8, 0.1])

        metrics = compute_outlier_metrics(y_true, y_pred, scores)

        assert "roc_auc" in metrics


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_classification_evaluation(self):
        """Test evaluating a classification model."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)

        result = evaluate_model(model, X_train, y_train, X_test, y_test, "classification")

        assert result["task_type"] == "classification"
        assert "train_metrics" in result
        assert "test_metrics" in result
        assert "accuracy" in result["test_metrics"]
        assert "confusion_matrix" in result

    def test_regression_evaluation(self):
        """Test evaluating a regression model."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        result = evaluate_model(model, X_train, y_train, X_test, y_test, "regression")

        assert result["task_type"] == "regression"
        assert "r2" in result["test_metrics"]
        assert "residuals" in result

    def test_clustering_evaluation(self):
        """Test evaluating a clustering model."""
        X = np.random.randn(100, 5)
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

        model = KMeans(n_clusters=3, random_state=42, n_init=10)
        model.fit(X_train)

        result = evaluate_model(model, X_train, None, X_test, None, "clustering")

        assert result["task_type"] == "clustering"
        assert "labels" in result


class TestRunCrossValidation:
    """Tests for run_cross_validation function."""

    def test_classification_cv(self):
        """Test cross-validation for classification."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = LogisticRegression(random_state=42)

        result = run_cross_validation(model, X, y, cv=3, task_type="classification")

        assert "test_accuracy" in result or "error" not in result

    def test_regression_cv(self):
        """Test cross-validation for regression."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        model = LinearRegression()

        result = run_cross_validation(model, X, y, cv=3, task_type="regression")

        assert "test_r2" in result or "error" not in result


class TestPlotFunctions:
    """Tests for plotting functions."""

    def test_plot_confusion_matrix(self):
        """Test confusion matrix plotting."""
        cm = np.array([[50, 10], [5, 35]])
        fig = plot_confusion_matrix(cm)
        assert fig is not None

    def test_plot_confusion_matrix_with_labels(self):
        """Test confusion matrix with labels."""
        cm = np.array([[50, 10], [5, 35]])
        fig = plot_confusion_matrix(cm, labels=["Class A", "Class B"])
        assert fig is not None

    def test_plot_roc_curve(self):
        """Test ROC curve plotting."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_proba = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.6, 0.4, 0.9])

        fig = plot_roc_curve(y_true, y_proba)
        assert fig is not None

    def test_plot_precision_recall_curve(self):
        """Test precision-recall curve plotting."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_proba = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.6, 0.4, 0.9])

        fig = plot_precision_recall_curve(y_true, y_proba)
        assert fig is not None

    def test_plot_residuals(self):
        """Test residual plotting."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

        fig = plot_residuals(y_true, y_pred)
        assert fig is not None

    def test_plot_actual_vs_predicted(self):
        """Test actual vs predicted plotting."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

        fig = plot_actual_vs_predicted(y_true, y_pred)
        assert fig is not None

    def test_plot_learning_curve(self):
        """Test learning curve plotting."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = LogisticRegression(random_state=42)

        fig = plot_learning_curve(model, X, y, cv=3)
        assert fig is not None

    def test_plot_feature_importance(self):
        """Test feature importance plotting."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        fig = plot_feature_importance(model, [f"feature_{i}" for i in range(5)])
        assert fig is not None

    def test_plot_cluster_visualization(self):
        """Test cluster visualization plotting."""
        X = np.random.randn(100, 5)
        labels = np.array([0] * 50 + [1] * 50)

        fig = plot_cluster_visualization(X, labels)
        assert fig is not None

    def test_plot_outlier_scores(self):
        """Test outlier scores plotting."""
        scores = np.random.randn(100)

        fig = plot_outlier_scores(scores)
        assert fig is not None

    def test_plot_outlier_scores_with_labels(self):
        """Test outlier scores with true labels."""
        scores = np.concatenate([np.random.randn(80) - 1, np.random.randn(20) + 2])
        y_true = np.array([0] * 80 + [1] * 20)

        fig = plot_outlier_scores(scores, y_true)
        assert fig is not None
