"""Hyperparameter tuning utilities.

Provides grid search, random search, and successive halving helpers.
"""

from typing import Any, Literal

import numpy as np
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
)

try:
    from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

    HALVING_AVAILABLE = True
except ImportError:
    HALVING_AVAILABLE = False

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from skplay.core.datasets import TaskType

# Parameter grids for common estimators
PARAM_GRIDS = {
    # Classification
    "LogisticRegression": {
        "beginner": {"C": [0.1, 1.0, 10.0]},
        "intermediate": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l2"],
            "solver": ["lbfgs", "saga"],
        },
        "advanced": {
            "C": np.logspace(-3, 3, 7).tolist(),
            "penalty": ["l1", "l2", "elasticnet", None],
            "solver": ["lbfgs", "liblinear", "saga"],
            "max_iter": [100, 500, 1000],
        },
    },
    "RandomForestClassifier": {
        "beginner": {"n_estimators": [50, 100, 200]},
        "intermediate": {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
        },
        "advanced": {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [None, 5, 10, 20, 30, 50],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
        },
    },
    "SVC": {
        "beginner": {"C": [0.1, 1.0, 10.0]},
        "intermediate": {
            "C": [0.1, 1.0, 10.0, 100.0],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },
        "advanced": {
            "C": np.logspace(-2, 3, 6).tolist(),
            "kernel": ["rbf", "linear", "poly", "sigmoid"],
            "gamma": ["scale", "auto"] + np.logspace(-4, 1, 6).tolist(),
            "degree": [2, 3, 4],
        },
    },
    "GradientBoostingClassifier": {
        "beginner": {"n_estimators": [50, 100, 200]},
        "intermediate": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        },
        "advanced": {
            "n_estimators": [50, 100, 200, 500],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7, 10],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "min_samples_split": [2, 5, 10],
        },
    },
    # Regression
    "Ridge": {
        "beginner": {"alpha": [0.1, 1.0, 10.0]},
        "intermediate": {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        "advanced": {
            "alpha": np.logspace(-3, 3, 7).tolist(),
            "solver": ["auto", "svd", "cholesky", "lsqr", "saga"],
        },
    },
    "Lasso": {
        "beginner": {"alpha": [0.1, 1.0, 10.0]},
        "intermediate": {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        },
        "advanced": {
            "alpha": np.logspace(-4, 2, 7).tolist(),
            "max_iter": [1000, 5000, 10000],
        },
    },
    "RandomForestRegressor": {
        "beginner": {"n_estimators": [50, 100, 200]},
        "intermediate": {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
        },
        "advanced": {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [None, 5, 10, 20, 30, 50],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", None],
        },
    },
    # Clustering
    "KMeans": {
        "beginner": {"n_clusters": [2, 3, 4, 5]},
        "intermediate": {
            "n_clusters": list(range(2, 11)),
            "init": ["k-means++", "random"],
        },
        "advanced": {
            "n_clusters": list(range(2, 16)),
            "init": ["k-means++", "random"],
            "n_init": [10, 20, 50],
            "max_iter": [300, 500, 1000],
        },
    },
    "DBSCAN": {
        "beginner": {"eps": [0.3, 0.5, 1.0]},
        "intermediate": {
            "eps": [0.1, 0.3, 0.5, 0.7, 1.0],
            "min_samples": [3, 5, 10],
        },
        "advanced": {
            "eps": np.linspace(0.1, 2.0, 10).tolist(),
            "min_samples": [3, 5, 10, 15, 20],
            "metric": ["euclidean", "manhattan", "cosine"],
        },
    },
    # Outlier detection
    "IsolationForest": {
        "beginner": {"contamination": [0.01, 0.05, 0.1]},
        "intermediate": {
            "n_estimators": [50, 100, 200],
            "contamination": [0.01, 0.05, 0.1, 0.2],
            "max_samples": ["auto", 0.5, 0.8],
        },
        "advanced": {
            "n_estimators": [50, 100, 200, 500],
            "contamination": [0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
            "max_samples": ["auto", 0.3, 0.5, 0.7, 1.0],
            "max_features": [0.5, 0.7, 1.0],
        },
    },
}


def get_param_grid(
    estimator_name: str,
    level: Literal["beginner", "intermediate", "advanced"] = "intermediate",
) -> dict:
    """Get parameter grid for an estimator at a given level.

    Args:
        estimator_name: Name of the estimator class
        level: User expertise level

    Returns:
        Parameter grid dictionary
    """
    if estimator_name in PARAM_GRIDS:
        level_grids = PARAM_GRIDS[estimator_name]
        if isinstance(level_grids, dict):
            return level_grids.get(level, {})
    return {}


def get_random_param_distributions(
    estimator_name: str,
    level: Literal["beginner", "intermediate", "advanced"] = "intermediate",
) -> dict:
    """Get parameter distributions for random search.

    For now, returns the same as param_grid (discrete values).
    Could be extended with scipy distributions for continuous params.
    """
    return get_param_grid(estimator_name, level)


def run_grid_search(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    param_grid: dict,
    cv: int = 5,
    scoring: str | None = None,
    n_jobs: int = -1,
    verbose: int = 0,
) -> GridSearchCV:
    """Run grid search cross-validation.

    Args:
        pipeline: The pipeline to tune
        X: Feature matrix
        y: Target vector
        param_grid: Parameter grid (keys should include step name prefix)
        cv: Number of CV folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs
        verbose: Verbosity level

    Returns:
        Fitted GridSearchCV object
    """
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
        return_train_score=True,
    )
    grid_search.fit(X, y)
    return grid_search


def run_random_search(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    param_distributions: dict,
    n_iter: int = 20,
    cv: int = 5,
    scoring: str | None = None,
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: int = 0,
) -> RandomizedSearchCV:
    """Run randomized search cross-validation.

    Args:
        pipeline: The pipeline to tune
        X: Feature matrix
        y: Target vector
        param_distributions: Parameter distributions
        n_iter: Number of parameter settings sampled
        cv: Number of CV folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs
        random_state: Random seed
        verbose: Verbosity level

    Returns:
        Fitted RandomizedSearchCV object
    """
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
        refit=True,
        return_train_score=True,
    )
    random_search.fit(X, y)
    return random_search


def run_halving_search(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    param_grid: dict,
    factor: int = 3,
    cv: int = 5,
    scoring: str | None = None,
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: int = 0,
    use_random: bool = False,
) -> Any:
    """Run successive halving search (if available).

    Args:
        pipeline: The pipeline to tune
        X: Feature matrix
        y: Target vector
        param_grid: Parameter grid
        factor: Proportion of candidates that are selected for each round
        cv: Number of CV folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs
        random_state: Random seed
        verbose: Verbosity level
        use_random: Use HalvingRandomSearchCV instead of HalvingGridSearchCV

    Returns:
        Fitted search object or raises ImportError if not available
    """
    if not HALVING_AVAILABLE:
        raise ImportError("Successive halving not available. Upgrade to scikit-learn >= 1.0.")

    if use_random:
        search = HalvingRandomSearchCV(
            pipeline,
            param_grid,
            factor=factor,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            refit=True,
        )
    else:
        search = HalvingGridSearchCV(
            pipeline,
            param_grid,
            factor=factor,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            refit=True,
        )

    search.fit(X, y)
    return search


def quick_cv_score(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str | None = None,
) -> tuple[float, float]:
    """Quick cross-validation scoring.

    Args:
        estimator: Estimator to evaluate
        X: Feature matrix
        y: Target vector
        cv: Number of CV folds
        scoring: Scoring metric

    Returns:
        Tuple of (mean_score, std_score)
    """
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
    return float(scores.mean()), float(scores.std())


def format_search_results(search_cv: Any, top_n: int = 5) -> list[dict]:
    """Format search CV results for display.

    Args:
        search_cv: Fitted GridSearchCV or RandomizedSearchCV
        top_n: Number of top results to return

    Returns:
        List of result dictionaries
    """
    results = []
    cv_results = search_cv.cv_results_

    # Get indices sorted by rank
    ranks = cv_results["rank_test_score"]
    top_indices = np.argsort(ranks)[:top_n]

    for idx in top_indices:
        result = {
            "rank": int(ranks[idx]),
            "params": cv_results["params"][idx],
            "mean_test_score": float(cv_results["mean_test_score"][idx]),
            "std_test_score": float(cv_results["std_test_score"][idx]),
        }
        if "mean_train_score" in cv_results:
            result["mean_train_score"] = float(cv_results["mean_train_score"][idx])
        results.append(result)

    return results


def suggest_param_ranges(
    estimator_name: str,
    task_type: TaskType,
) -> dict:
    """Suggest sensible parameter ranges for an estimator.

    Args:
        estimator_name: Name of the estimator class
        task_type: The task type

    Returns:
        Dictionary of parameter name -> (min, max, default, type)
    """
    # Common suggestions
    suggestions = {
        "n_estimators": (10, 1000, 100, "int"),
        "max_depth": (1, 50, None, "int_or_none"),
        "learning_rate": (0.001, 1.0, 0.1, "float"),
        "C": (0.001, 1000, 1.0, "float_log"),
        "alpha": (0.0001, 100, 1.0, "float_log"),
        "gamma": (0.0001, 10, "scale", "float_log_or_str"),
        "min_samples_split": (2, 50, 2, "int"),
        "min_samples_leaf": (1, 50, 1, "int"),
        "n_neighbors": (1, 50, 5, "int"),
        "n_clusters": (2, 20, 3, "int"),
        "eps": (0.01, 5.0, 0.5, "float"),
        "contamination": (0.001, 0.5, 0.1, "float"),
    }

    # Get default params for estimator
    from skplay.core.estimators import get_estimator_class

    est_class = get_estimator_class(task_type, estimator_name)

    if not est_class:
        return {}

    try:
        default_params = est_class().get_params()
    except Exception:
        return {}

    result = {}
    for param_name, default_value in default_params.items():
        if param_name in suggestions:
            min_val, max_val, suggested_default, param_type = suggestions[param_name]
            result[param_name] = {
                "min": min_val,
                "max": max_val,
                "default": default_value,
                "suggested_default": suggested_default,
                "type": param_type,
            }

    return result
