"""Evaluation utilities for model assessment.

Provides metrics computation, cross-validation, and visualization functions.
"""

from typing import Any, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    cross_validate,
    learning_curve,
    validation_curve,
    train_test_split,
)
from sklearn.metrics import (
    # Classification
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    # Regression
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    # Clustering
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.inspection import permutation_importance

try:
    from sklearn.inspection import PartialDependenceDisplay
    PDP_AVAILABLE = True
except ImportError:
    PDP_AVAILABLE = False

from skplay.core.datasets import TaskType


def get_metrics_for_task(task_type: TaskType) -> list[str]:
    """Get appropriate metrics for a task type.

    Args:
        task_type: The task type

    Returns:
        List of metric names
    """
    metrics = {
        "classification": ["accuracy", "precision", "recall", "f1", "roc_auc"],
        "regression": ["r2", "mae", "rmse", "mape"],
        "clustering": ["silhouette", "calinski_harabasz", "davies_bouldin"],
        "outlier_detection": ["precision", "recall", "f1", "roc_auc"],
    }
    return metrics.get(task_type, [])


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    average: str = "weighted",
) -> dict[str, float]:
    """Compute classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        average: Averaging method for multiclass

    Returns:
        Dictionary of metric name -> value
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    # ROC AUC if probabilities available
    if y_proba is not None:
        try:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                # Binary classification
                if y_proba.ndim == 2:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
                metrics["pr_auc"] = average_precision_score(
                    y_true, y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                )
            else:
                # Multiclass
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average=average
                )
        except Exception:
            pass  # Some metrics may not be computable

    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of metric name -> value
    """
    metrics = {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }

    # MAPE (handle zeros)
    try:
        metrics["mape"] = mean_absolute_percentage_error(y_true, y_pred)
    except Exception:
        metrics["mape"] = np.nan

    return metrics


def compute_clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Compute clustering metrics.

    Args:
        X: Feature matrix
        labels: Cluster labels

    Returns:
        Dictionary of metric name -> value
    """
    n_clusters = len(np.unique(labels[labels >= 0]))

    metrics = {}

    if n_clusters > 1:
        try:
            metrics["silhouette"] = silhouette_score(X, labels)
        except Exception:
            metrics["silhouette"] = np.nan

        try:
            metrics["calinski_harabasz"] = calinski_harabasz_score(X, labels)
        except Exception:
            metrics["calinski_harabasz"] = np.nan

        try:
            metrics["davies_bouldin"] = davies_bouldin_score(X, labels)
        except Exception:
            metrics["davies_bouldin"] = np.nan

    metrics["n_clusters"] = n_clusters

    return metrics


def compute_outlier_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute outlier detection metrics (if labels available).

    Args:
        y_true: True labels (1 for outlier, 0 for normal)
        y_pred: Predicted labels
        scores: Anomaly scores (optional)

    Returns:
        Dictionary of metric name -> value
    """
    # Convert predictions to binary (outlier detection models return -1/1)
    y_pred_binary = (y_pred == -1).astype(int) if -1 in y_pred else y_pred
    y_true_binary = (y_true == 1).astype(int) if isinstance(y_true[0], (int, np.integer)) else y_true

    metrics = {
        "precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
        "recall": recall_score(y_true_binary, y_pred_binary, zero_division=0),
        "f1": f1_score(y_true_binary, y_pred_binary, zero_division=0),
    }

    if scores is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true_binary, -scores)  # Negate scores
            metrics["pr_auc"] = average_precision_score(y_true_binary, -scores)
        except Exception:
            pass

    return metrics


def evaluate_model(
    model: BaseEstimator | Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray | None,
    X_test: np.ndarray,
    y_test: np.ndarray | None,
    task_type: TaskType,
) -> dict[str, Any]:
    """Evaluate a fitted model on train and test sets.

    Args:
        model: Fitted model or pipeline
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        task_type: The task type

    Returns:
        Dictionary with train/test metrics and predictions
    """
    result = {
        "task_type": task_type,
        "train_metrics": {},
        "test_metrics": {},
    }

    if task_type == "classification":
        # Training metrics
        y_train_pred = model.predict(X_train)
        y_train_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_train_proba = model.predict_proba(X_train)
            except Exception:
                pass
        result["train_metrics"] = compute_classification_metrics(
            y_train, y_train_pred, y_train_proba
        )

        # Test metrics
        y_test_pred = model.predict(X_test)
        y_test_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_test_proba = model.predict_proba(X_test)
            except Exception:
                pass
        result["test_metrics"] = compute_classification_metrics(
            y_test, y_test_pred, y_test_proba
        )

        result["y_pred"] = y_test_pred
        result["y_proba"] = y_test_proba
        result["confusion_matrix"] = confusion_matrix(y_test, y_test_pred)

    elif task_type == "regression":
        # Training metrics
        y_train_pred = model.predict(X_train)
        result["train_metrics"] = compute_regression_metrics(y_train, y_train_pred)

        # Test metrics
        y_test_pred = model.predict(X_test)
        result["test_metrics"] = compute_regression_metrics(y_test, y_test_pred)

        result["y_pred"] = y_test_pred
        result["residuals"] = y_test - y_test_pred

    elif task_type == "clustering":
        # Clustering only evaluated on features
        labels = model.predict(X_test) if hasattr(model, "predict") else model.labels_
        result["test_metrics"] = compute_clustering_metrics(X_test, labels)
        result["labels"] = labels

        # Also compute on training set
        train_labels = model.predict(X_train) if hasattr(model, "predict") else model.labels_
        result["train_metrics"] = compute_clustering_metrics(X_train, train_labels)

    elif task_type == "outlier_detection":
        y_pred = model.predict(X_test)
        scores = None
        if hasattr(model, "decision_function"):
            try:
                scores = model.decision_function(X_test)
            except Exception:
                pass
        elif hasattr(model, "score_samples"):
            try:
                scores = model.score_samples(X_test)
            except Exception:
                pass

        if y_test is not None:
            # Convert string labels to binary if needed
            if y_test.dtype == object:
                # Assume 'fraud', 'outlier', 'anomaly' etc. are the positive class
                positive_labels = {'fraud', 'outlier', 'anomaly', '1', 'true', 'yes'}
                y_test_binary = np.array([1 if str(y).lower() in positive_labels else 0 for y in y_test])
            else:
                y_test_binary = y_test

            result["test_metrics"] = compute_outlier_metrics(y_test_binary, y_pred, scores)

        result["y_pred"] = y_pred
        result["scores"] = scores

    return result


def run_cross_validation(
    model: BaseEstimator | Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: list[str] | None = None,
    task_type: TaskType = "classification",
) -> dict[str, Any]:
    """Run cross-validation and return results.

    Args:
        model: Model or pipeline to evaluate
        X: Feature matrix
        y: Target vector
        cv: Number of folds
        scoring: List of scoring metrics
        task_type: The task type

    Returns:
        Cross-validation results dictionary
    """
    if scoring is None:
        if task_type == "classification":
            scoring = ["accuracy", "f1_weighted", "roc_auc_ovr_weighted"]
        elif task_type == "regression":
            scoring = ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]
        else:
            scoring = ["accuracy"]

    try:
        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            return_estimator=False,
        )

        # Format results
        formatted = {}
        for key, values in cv_results.items():
            if key.startswith("test_") or key.startswith("train_"):
                formatted[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "values": values.tolist(),
                }
            else:
                formatted[key] = values.tolist() if isinstance(values, np.ndarray) else values

        return formatted
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str] | None = None,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Plot confusion matrix heatmap.

    Args:
        cm: Confusion matrix
        labels: Class labels
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    if labels:
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=labels,
            yticklabels=labels,
        )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    fig.tight_layout()

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "ROC Curve",
) -> plt.Figure:
    """Plot ROC curve for binary classification.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if y_proba.ndim == 2:
        y_score = y_proba[:, 1]
    else:
        y_score = y_proba

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "Precision-Recall Curve",
) -> plt.Figure:
    """Plot precision-recall curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if y_proba.ndim == 2:
        y_score = y_proba[:, 1]
    else:
        y_score = y_proba

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    ax.plot(recall, precision, 'b-', linewidth=2, label=f'PR (AP = {ap:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='lower left')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Plot",
) -> plt.Figure:
    """Plot residuals for regression.

    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    residuals = y_true - y_pred

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Predicted')
    axes[0].grid(True, alpha=0.3)

    # Residual histogram
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Residual Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()

    return fig


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted",
) -> plt.Figure:
    """Plot actual vs predicted values for regression.

    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_learning_curve(
    model: BaseEstimator | Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    train_sizes: np.ndarray | None = None,
    scoring: str | None = None,
    title: str = "Learning Curve",
) -> plt.Figure:
    """Plot learning curve showing train/validation scores vs training size.

    Args:
        model: Model or pipeline
        X: Feature matrix
        y: Target vector
        cv: Number of CV folds
        train_sizes: Training set sizes to evaluate
        scoring: Scoring metric
        title: Plot title

    Returns:
        Matplotlib figure
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes, scoring=scoring, n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                    alpha=0.1, color='blue')
    ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std,
                    alpha=0.1, color='orange')
    ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training score')
    ax.plot(train_sizes_abs, test_mean, 'o-', color='orange', label='Validation score')

    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_validation_curve(
    model: BaseEstimator | Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    param_name: str,
    param_range: np.ndarray,
    cv: int = 5,
    scoring: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot validation curve for a hyperparameter.

    Args:
        model: Model or pipeline
        X: Feature matrix
        y: Target vector
        param_name: Parameter to vary
        param_range: Values to test
        cv: Number of CV folds
        scoring: Scoring metric
        title: Plot title

    Returns:
        Matplotlib figure
    """
    train_scores, test_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                    alpha=0.1, color='blue')
    ax.fill_between(param_range, test_mean - test_std, test_mean + test_std,
                    alpha=0.1, color='orange')
    ax.plot(param_range, train_mean, 'o-', color='blue', label='Training score')
    ax.plot(param_range, test_mean, 'o-', color='orange', label='Validation score')

    ax.set_xlabel(param_name)
    ax.set_ylabel('Score')
    ax.set_title(title or f'Validation Curve: {param_name}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Use log scale if range spans orders of magnitude
    if param_range.max() / param_range.min() > 100:
        ax.set_xscale('log')

    fig.tight_layout()

    return fig


def plot_feature_importance(
    model: BaseEstimator | Pipeline,
    feature_names: list[str],
    top_n: int = 20,
    title: str = "Feature Importance",
) -> plt.Figure:
    """Plot feature importance (for tree-based models).

    Args:
        model: Fitted model with feature_importances_
        feature_names: Feature names
        top_n: Number of top features to show
        title: Plot title

    Returns:
        Matplotlib figure
    """
    # Get feature importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps"):
        # Pipeline - try to get from last step
        estimator = model.named_steps.get("estimator") or list(model.named_steps.values())[-1]
        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        else:
            raise ValueError("Model does not have feature_importances_")
    else:
        raise ValueError("Model does not have feature_importances_")

    # Sort and select top N
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, max(6, len(indices) * 0.3)))

    ax.barh(range(len(indices)), importances[indices][::-1], align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()

    return fig


def plot_permutation_importance(
    model: BaseEstimator | Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 10,
    top_n: int = 20,
    scoring: str | None = None,
    title: str = "Permutation Importance",
) -> plt.Figure:
    """Plot permutation importance.

    Args:
        model: Fitted model
        X: Feature matrix
        y: Target vector
        feature_names: Feature names
        n_repeats: Number of permutation repeats
        top_n: Number of top features to show
        scoring: Scoring metric
        title: Plot title

    Returns:
        Matplotlib figure
    """
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, scoring=scoring, n_jobs=-1
    )

    # Sort by importance
    sorted_idx = result.importances_mean.argsort()[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_idx) * 0.3)))

    ax.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=[feature_names[i] for i in sorted_idx]
    )
    ax.set_xlabel('Decrease in Score')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()

    return fig


def plot_cluster_visualization(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str] | None = None,
    method: Literal["pca", "first_two"] = "pca",
    title: str = "Cluster Visualization",
) -> plt.Figure:
    """Visualize clusters in 2D.

    Args:
        X: Feature matrix
        labels: Cluster labels
        feature_names: Feature names (for axis labels)
        method: Dimensionality reduction method
        title: Plot title

    Returns:
        Matplotlib figure
    """
    from sklearn.decomposition import PCA

    fig, ax = plt.subplots(figsize=(10, 8))

    if X.shape[1] > 2 and method == "pca":
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        x_label = f"PC1 ({pca.explained_variance_ratio_[0]:.1%})"
        y_label = f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"
    else:
        X_2d = X[:, :2]
        if feature_names:
            x_label = feature_names[0]
            y_label = feature_names[1] if len(feature_names) > 1 else "Feature 2"
        else:
            x_label = "Feature 1"
            y_label = "Feature 2"

    scatter = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=labels, cmap='viridis', alpha=0.6, edgecolors='k', linewidth=0.5
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster')

    fig.tight_layout()

    return fig


def plot_outlier_scores(
    scores: np.ndarray,
    y_true: np.ndarray | None = None,
    threshold: float | None = None,
    title: str = "Outlier Scores Distribution",
) -> plt.Figure:
    """Plot outlier score distribution.

    Args:
        scores: Anomaly scores
        y_true: True labels (if available)
        threshold: Decision threshold
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if y_true is not None:
        # Separate scores by class
        # Convert string labels if needed
        if y_true.dtype == object:
            positive_labels = {'fraud', 'outlier', 'anomaly', '1', 'true', 'yes'}
            y_binary = np.array([1 if str(y).lower() in positive_labels else 0 for y in y_true])
        else:
            y_binary = y_true

        normal_scores = scores[y_binary == 0]
        outlier_scores = scores[y_binary == 1]

        ax.hist(normal_scores, bins=30, alpha=0.5, label='Normal', color='blue')
        ax.hist(outlier_scores, bins=30, alpha=0.5, label='Outlier', color='red')
        ax.legend()
    else:
        ax.hist(scores, bins=50, edgecolor='black', alpha=0.7)

    if threshold is not None:
        ax.axvline(x=threshold, color='k', linestyle='--', linewidth=2, label='Threshold')

    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig
