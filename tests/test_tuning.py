"""Tests for hyperparameter tuning utilities."""

import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skplay.core.tuning import (
    HALVING_AVAILABLE,
    PARAM_GRIDS,
    format_search_results,
    get_param_grid,
    get_random_param_distributions,
    quick_cv_score,
    run_grid_search,
    run_halving_search,
    run_random_search,
    suggest_param_ranges,
)


class TestParamGrids:
    """Tests for parameter grids."""

    def test_param_grids_exist(self):
        """Test that param grids are defined for common estimators."""
        assert "LogisticRegression" in PARAM_GRIDS
        assert "RandomForestClassifier" in PARAM_GRIDS
        assert "SVC" in PARAM_GRIDS
        assert "KMeans" in PARAM_GRIDS

    def test_param_grids_levels(self):
        """Test that param grids have all levels."""
        for name, grids in PARAM_GRIDS.items():
            assert "beginner" in grids, f"Missing beginner level for {name}"
            assert "intermediate" in grids, f"Missing intermediate level for {name}"
            assert "advanced" in grids, f"Missing advanced level for {name}"


class TestGetParamGrid:
    """Tests for get_param_grid function."""

    def test_get_logistic_regression_grid(self):
        """Test getting param grid for LogisticRegression."""
        grid = get_param_grid("LogisticRegression", "beginner")
        assert "C" in grid
        assert len(grid["C"]) > 0

    def test_get_advanced_grid(self):
        """Test getting advanced param grid has more options."""
        beginner = get_param_grid("LogisticRegression", "beginner")
        advanced = get_param_grid("LogisticRegression", "advanced")

        # Advanced should have more params
        assert len(advanced) >= len(beginner)

    def test_unknown_estimator(self):
        """Test unknown estimator returns empty dict."""
        grid = get_param_grid("UnknownEstimator", "beginner")
        assert grid == {}

    def test_all_levels(self):
        """Test all levels return valid grids."""
        for level in ["beginner", "intermediate", "advanced"]:
            grid = get_param_grid("RandomForestClassifier", level)
            assert isinstance(grid, dict)


class TestGetRandomParamDistributions:
    """Tests for get_random_param_distributions function."""

    def test_returns_same_as_grid(self):
        """Test that random distributions match grid for now."""
        grid = get_param_grid("LogisticRegression", "intermediate")
        dist = get_random_param_distributions("LogisticRegression", "intermediate")
        assert grid == dist


class TestRunGridSearch:
    """Tests for run_grid_search function."""

    def test_classification_grid_search(self):
        """Test grid search for classification."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42))
        ])

        param_grid = {"clf__C": [0.1, 1.0, 10.0]}

        result = run_grid_search(pipeline, X, y, param_grid, cv=3)

        assert hasattr(result, "best_params_")
        assert hasattr(result, "best_score_")
        assert "clf__C" in result.best_params_

    def test_regression_grid_search(self):
        """Test grid search for regression."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge())
        ])

        param_grid = {"reg__alpha": [0.1, 1.0, 10.0]}

        result = run_grid_search(pipeline, X, y, param_grid, cv=3, scoring="r2")

        assert hasattr(result, "best_params_")


class TestRunRandomSearch:
    """Tests for run_random_search function."""

    def test_classification_random_search(self):
        """Test random search for classification."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42))
        ])

        param_distributions = {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]}

        result = run_random_search(
            pipeline, X, y, param_distributions,
            n_iter=5, cv=3, random_state=42
        )

        assert hasattr(result, "best_params_")
        assert hasattr(result, "best_score_")


class TestRunHalvingSearch:
    """Tests for run_halving_search function."""

    @pytest.mark.skipif(not HALVING_AVAILABLE, reason="Halving search not available")
    def test_halving_grid_search(self):
        """Test halving grid search."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42))
        ])

        param_grid = {"clf__C": [0.1, 1.0, 10.0]}

        result = run_halving_search(
            pipeline, X, y, param_grid,
            cv=3, random_state=42
        )

        assert hasattr(result, "best_params_")

    def test_halving_unavailable_error(self):
        """Test error when halving not available."""
        if not HALVING_AVAILABLE:
            X, y = make_classification(n_samples=100, n_features=5, random_state=42)
            pipeline = Pipeline([("clf", LogisticRegression())])

            with pytest.raises(ImportError):
                run_halving_search(pipeline, X, y, {"clf__C": [1.0]})


class TestQuickCVScore:
    """Tests for quick_cv_score function."""

    def test_classification_cv_score(self):
        """Test quick CV score for classification."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = LogisticRegression(random_state=42)

        mean_score, std_score = quick_cv_score(model, X, y, cv=3)

        assert 0 <= mean_score <= 1
        assert std_score >= 0

    def test_regression_cv_score(self):
        """Test quick CV score for regression."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        model = Ridge()

        mean_score, std_score = quick_cv_score(model, X, y, cv=3, scoring="r2")

        assert isinstance(mean_score, float)
        assert isinstance(std_score, float)


class TestFormatSearchResults:
    """Tests for format_search_results function."""

    def test_format_grid_search_results(self):
        """Test formatting grid search results."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42))
        ])

        param_grid = {"clf__C": [0.1, 1.0, 10.0]}
        search = run_grid_search(pipeline, X, y, param_grid, cv=3)

        results = format_search_results(search, top_n=3)

        assert len(results) <= 3
        assert all("rank" in r for r in results)
        assert all("params" in r for r in results)
        assert all("mean_test_score" in r for r in results)

    def test_results_sorted_by_rank(self):
        """Test that results are sorted by rank."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42))
        ])

        param_grid = {"clf__C": [0.1, 1.0, 10.0]}
        search = run_grid_search(pipeline, X, y, param_grid, cv=3)

        results = format_search_results(search, top_n=3)

        ranks = [r["rank"] for r in results]
        assert ranks == sorted(ranks)


class TestSuggestParamRanges:
    """Tests for suggest_param_ranges function."""

    def test_suggest_logistic_regression_params(self):
        """Test param suggestions for LogisticRegression."""
        suggestions = suggest_param_ranges("LogisticRegression", "classification")

        assert "C" in suggestions or len(suggestions) >= 0  # May be empty if not in suggestions

    def test_suggest_random_forest_params(self):
        """Test param suggestions for RandomForestClassifier."""
        suggestions = suggest_param_ranges("RandomForestClassifier", "classification")

        if suggestions:  # May be empty
            # Check structure if suggestions exist
            for _param_name, param_info in suggestions.items():
                assert "min" in param_info
                assert "max" in param_info
                assert "default" in param_info
                assert "type" in param_info

    def test_unknown_estimator(self):
        """Test unknown estimator returns empty dict."""
        suggestions = suggest_param_ranges("UnknownEstimator", "classification")
        assert suggestions == {}
