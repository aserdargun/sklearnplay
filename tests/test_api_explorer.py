"""Tests for API Explorer utilities."""

from sklearn.linear_model import LogisticRegression

from skplay.core.api_explorer import (
    APIExplorer,
    build_api_entry,
    generate_example_snippet,
    get_api_entry,
    get_estimator_type,
    get_explorer,
    get_parameter_info,
    get_short_docstring,
    get_signature_string,
    get_tags_from_module,
    search_api,
)


class TestGetShortDocstring:
    """Tests for get_short_docstring function."""

    def test_extract_docstring(self):
        """Test extracting short docstring."""

        class TestClass:
            """This is a test class.

            With more details here.
            """
            pass

        doc = get_short_docstring(TestClass)
        assert "test class" in doc.lower()

    def test_no_docstring(self):
        """Test handling class without docstring."""

        class NoDoc:
            pass

        doc = get_short_docstring(NoDoc)
        assert doc == ""

    def test_sklearn_estimator(self):
        """Test extracting docstring from sklearn estimator."""
        doc = get_short_docstring(LogisticRegression)
        assert len(doc) > 0


class TestGetSignatureString:
    """Tests for get_signature_string function."""

    def test_function_signature(self):
        """Test getting function signature."""

        def test_func(a, b, c=10):
            pass

        sig = get_signature_string(test_func)
        assert "a" in sig
        assert "b" in sig
        assert "c" in sig

    def test_class_signature(self):
        """Test getting class signature."""
        sig = get_signature_string(LogisticRegression)
        # Should have some parameters
        assert "(" in sig
        assert ")" in sig

    def test_invalid_signature(self):
        """Test handling object without signature."""
        sig = get_signature_string(42)  # Not callable
        assert sig == "()"


class TestGetParameterInfo:
    """Tests for get_parameter_info function."""

    def test_extract_params(self):
        """Test extracting parameter info."""

        def test_func(a, b=10, c="default"):
            """
            Parameters
            ----------
            a : int
                First parameter.
            b : int
                Second parameter.
            """
            pass

        params = get_parameter_info(test_func)

        assert len(params) >= 2
        assert any(p["name"] == "a" for p in params)
        assert any(p["name"] == "b" for p in params)

    def test_sklearn_estimator_params(self):
        """Test extracting params from sklearn estimator."""
        params = get_parameter_info(LogisticRegression)

        # LogisticRegression has C parameter
        param_names = [p["name"] for p in params]
        assert "C" in param_names or "penalty" in param_names


class TestGetEstimatorType:
    """Tests for get_estimator_type function."""

    def test_classifier(self):
        """Test identifying classifier."""
        est_type = get_estimator_type(LogisticRegression)
        assert est_type == "classifier"

    def test_regressor(self):
        """Test identifying regressor."""
        from sklearn.linear_model import Ridge

        est_type = get_estimator_type(Ridge)
        assert est_type == "regressor"

    def test_clusterer(self):
        """Test identifying clusterer."""
        from sklearn.cluster import KMeans

        est_type = get_estimator_type(KMeans)
        assert est_type == "clusterer"

    def test_transformer(self):
        """Test identifying transformer."""
        from sklearn.preprocessing import StandardScaler

        est_type = get_estimator_type(StandardScaler)
        assert est_type == "transformer"


class TestGetTagsFromModule:
    """Tests for get_tags_from_module function."""

    def test_linear_model_tags(self):
        """Test tags for linear_model module."""
        tags = get_tags_from_module("sklearn.linear_model")
        assert "linear" in tags

    def test_ensemble_tags(self):
        """Test tags for ensemble module."""
        tags = get_tags_from_module("sklearn.ensemble")
        assert "ensemble" in tags

    def test_preprocessing_tags(self):
        """Test tags for preprocessing module."""
        tags = get_tags_from_module("sklearn.preprocessing")
        assert "preprocessing" in tags

    def test_unknown_module(self):
        """Test unknown module returns empty tags."""
        tags = get_tags_from_module("unknown.module")
        assert tags == []


class TestBuildAPIEntry:
    """Tests for build_api_entry function."""

    def test_build_estimator_entry(self):
        """Test building entry for estimator."""
        entry = build_api_entry(
            "LogisticRegression",
            LogisticRegression,
            "sklearn.linear_model",
            "class"
        )

        assert entry.name == "LogisticRegression"
        assert entry.kind == "class"
        assert entry.is_estimator is True
        assert entry.estimator_type == "classifier"
        assert len(entry.parameters) > 0

    def test_build_function_entry(self):
        """Test building entry for function."""
        from sklearn.metrics import accuracy_score

        entry = build_api_entry(
            "accuracy_score",
            accuracy_score,
            "sklearn.metrics",
            "function"
        )

        assert entry.name == "accuracy_score"
        assert entry.kind == "function"
        assert entry.is_estimator is False


class TestAPIExplorer:
    """Tests for APIExplorer class."""

    def test_load_entries(self):
        """Test loading API entries."""
        explorer = APIExplorer()
        explorer.load()

        assert len(explorer._entries) > 0
        assert explorer._loaded is True

    def test_search(self):
        """Test searching API."""
        explorer = APIExplorer()
        results = explorer.search("logistic")

        assert len(results) > 0
        assert any("logistic" in r.name.lower() for r in results)

    def test_search_by_kind(self):
        """Test searching by kind."""
        explorer = APIExplorer()
        results = explorer.search("", kind="class", limit=100)

        assert all(r.kind == "class" for r in results)

    def test_search_by_estimator_type(self):
        """Test searching by estimator type."""
        explorer = APIExplorer()
        results = explorer.search("", estimator_type="classifier", limit=50)

        assert all(r.estimator_type == "classifier" for r in results)

    def test_get_entry(self):
        """Test getting specific entry."""
        explorer = APIExplorer()
        entry = explorer.get("LogisticRegression")

        assert entry is not None
        assert entry.name == "LogisticRegression"

    def test_get_nonexistent(self):
        """Test getting nonexistent entry."""
        explorer = APIExplorer()
        entry = explorer.get("NonexistentClass")

        assert entry is None

    def test_list_by_module(self):
        """Test listing by module."""
        explorer = APIExplorer()
        explorer.load()

        entries = explorer.list_by_module("sklearn.linear_model._logistic")

        # May have entries or not depending on exact module path
        assert isinstance(entries, list)

    def test_get_modules(self):
        """Test getting all modules."""
        explorer = APIExplorer()
        modules = explorer.get_modules()

        assert len(modules) > 0
        assert any("sklearn" in m for m in modules)

    def test_get_estimator_types(self):
        """Test getting estimator types."""
        explorer = APIExplorer()
        types = explorer.get_estimator_types()

        assert "classifier" in types
        assert "regressor" in types

    def test_get_all_names(self):
        """Test getting all entry names."""
        explorer = APIExplorer()
        names = explorer.get_all_names()

        assert len(names) > 0
        assert "LogisticRegression" in names

    def test_get_module_hierarchy(self):
        """Test getting module hierarchy."""
        explorer = APIExplorer()
        hierarchy = explorer.get_module_hierarchy()

        assert isinstance(hierarchy, dict)
        assert len(hierarchy) > 0


class TestGlobalExplorer:
    """Tests for global explorer functions."""

    def test_get_explorer_singleton(self):
        """Test explorer is singleton."""
        exp1 = get_explorer()
        exp2 = get_explorer()

        assert exp1 is exp2

    def test_search_api_function(self):
        """Test search_api convenience function."""
        results = search_api("random forest")

        assert len(results) > 0
        assert any("forest" in r.name.lower() for r in results)

    def test_get_api_entry_function(self):
        """Test get_api_entry convenience function."""
        entry = get_api_entry("StandardScaler")

        assert entry is not None
        assert entry.name == "StandardScaler"


class TestGenerateExampleSnippet:
    """Tests for generate_example_snippet function."""

    def test_estimator_snippet(self):
        """Test generating snippet for estimator."""
        entry = get_api_entry("LogisticRegression")
        assert entry is not None

        snippet = generate_example_snippet(entry)

        assert "LogisticRegression" in snippet
        assert "fit" in snippet
        assert "predict" in snippet

    def test_function_snippet(self):
        """Test generating snippet for function."""
        entry = get_api_entry("accuracy_score")
        assert entry is not None

        snippet = generate_example_snippet(entry)

        assert "accuracy_score" in snippet

    def test_snippet_has_import(self):
        """Test snippet includes import."""
        entry = get_api_entry("StandardScaler")
        assert entry is not None

        snippet = generate_example_snippet(entry)

        assert "from" in snippet
        assert "import" in snippet
