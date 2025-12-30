"""Tests for code snippet generation."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from skplay.core.snippets import (
    format_value,
    generate_code_snippet,
    generate_column_transformer_code,
    generate_estimator_code,
    generate_minimal_snippet,
    generate_pipeline_code,
    get_import_statement,
)


class TestGetImportStatement:
    """Tests for get_import_statement function."""

    def test_sklearn_estimator(self):
        """Test import statement for sklearn estimator."""
        stmt = get_import_statement(LogisticRegression)
        # Module path may be internal (e.g., sklearn.linear_model._logistic)
        assert "sklearn.linear_model" in stmt
        assert "import LogisticRegression" in stmt

    def test_sklearn_preprocessor(self):
        """Test import statement for sklearn preprocessor."""
        stmt = get_import_statement(StandardScaler)
        # Module path may be internal (e.g., sklearn.preprocessing._data)
        assert "sklearn.preprocessing" in stmt
        assert "import StandardScaler" in stmt


class TestFormatValue:
    """Tests for format_value function."""

    def test_none(self):
        """Test formatting None."""
        assert format_value(None) == "None"

    def test_string(self):
        """Test formatting string."""
        result = format_value("hello")
        assert result == "'hello'"

    def test_bool(self):
        """Test formatting boolean."""
        assert format_value(True) == "True"
        assert format_value(False) == "False"

    def test_int(self):
        """Test formatting integer."""
        assert format_value(42) == "42"

    def test_float(self):
        """Test formatting float."""
        assert format_value(3.14) == "3.14"

    def test_list(self):
        """Test formatting list."""
        result = format_value([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_tuple(self):
        """Test formatting tuple."""
        result = format_value((1, 2, 3))
        assert result == "(1, 2, 3)"

    def test_dict(self):
        """Test formatting dict."""
        result = format_value({"a": 1})
        assert "'a'" in result
        assert "1" in result

    def test_nested_structure(self):
        """Test formatting nested structure."""
        result = format_value([1, [2, 3], {"a": 4}])
        assert "[1, [2, 3]" in result


class TestGenerateEstimatorCode:
    """Tests for generate_estimator_code function."""

    def test_default_params(self):
        """Test code generation with default params."""
        model = LogisticRegression()
        imports, code = generate_estimator_code(model, "clf")

        assert "LogisticRegression" in code
        assert "clf = LogisticRegression()" in code
        assert any("LogisticRegression" in imp for imp in imports)

    def test_custom_params(self):
        """Test code generation with custom params."""
        model = LogisticRegression(C=0.5, max_iter=200)
        imports, code = generate_estimator_code(model, "clf")

        assert "C=0.5" in code
        assert "max_iter=200" in code

    def test_random_forest(self):
        """Test code generation for RandomForest."""
        model = RandomForestClassifier(n_estimators=50, max_depth=10)
        imports, code = generate_estimator_code(model, "rf")

        assert "RandomForestClassifier" in code
        assert "n_estimators=50" in code


class TestGeneratePipelineCode:
    """Tests for generate_pipeline_code function."""

    def test_simple_pipeline(self):
        """Test code generation for simple pipeline."""
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression())
        ])

        imports, code = generate_pipeline_code(pipeline)

        assert "Pipeline" in code
        assert "scaler" in code
        assert "clf" in code

    def test_pipeline_with_custom_params(self):
        """Test pipeline with custom parameters."""
        pipeline = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(C=0.1))
        ])

        imports, code = generate_pipeline_code(pipeline)

        assert "with_mean=False" in code
        assert "C=0.1" in code


class TestGenerateColumnTransformerCode:
    """Tests for generate_column_transformer_code function."""

    def test_simple_column_transformer(self):
        """Test code generation for simple ColumnTransformer."""
        ct = ColumnTransformer([
            ("num", StandardScaler(), [0, 1]),
            ("cat", OneHotEncoder(), [2])
        ])

        imports, code = generate_column_transformer_code(ct)

        assert "ColumnTransformer" in code
        assert "num" in code
        assert "cat" in code

    def test_with_passthrough(self):
        """Test ColumnTransformer with passthrough."""
        ct = ColumnTransformer([
            ("num", StandardScaler(), [0]),
            ("pass", "passthrough", [1])
        ])

        imports, code = generate_column_transformer_code(ct)

        assert "passthrough" in code

    def test_with_drop(self):
        """Test ColumnTransformer with drop."""
        ct = ColumnTransformer([
            ("num", StandardScaler(), [0]),
            ("drop", "drop", [1])
        ])

        imports, code = generate_column_transformer_code(ct)

        assert "drop" in code


class TestGenerateCodeSnippet:
    """Tests for generate_code_snippet function."""

    def test_classification_snippet(self):
        """Test full code snippet for classification."""
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression())
        ])

        snippet = generate_code_snippet(
            pipeline,
            X_train_shape=(80, 5),
            y_train_shape=(80,),
            dataset_name="test_data",
            task_type="classification"
        )

        assert "import" in snippet
        assert "train_test_split" in snippet
        assert "fit" in snippet
        assert "predict" in snippet
        assert "accuracy_score" in snippet

    def test_regression_snippet(self):
        """Test full code snippet for regression."""
        model = Ridge()

        snippet = generate_code_snippet(
            model,
            X_train_shape=(80, 5),
            y_train_shape=(80,),
            dataset_name="test_data",
            task_type="regression"
        )

        assert "r2_score" in snippet
        assert "mean_squared_error" in snippet

    def test_clustering_snippet(self):
        """Test full code snippet for clustering."""
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=3)

        snippet = generate_code_snippet(
            model,
            X_train_shape=(80, 5),
            y_train_shape=None,
            dataset_name="test_data",
            task_type="clustering"
        )

        assert "silhouette_score" in snippet

    def test_with_column_info(self):
        """Test snippet with column information."""
        pipeline = Pipeline([
            ("clf", LogisticRegression())
        ])

        snippet = generate_code_snippet(
            pipeline,
            X_train_shape=(80, 5),
            y_train_shape=(80,),
            numeric_cols=["a", "b", "c"],
            categorical_cols=["d", "e"],
            task_type="classification"
        )

        assert "numeric_cols" in snippet
        assert "categorical_cols" in snippet


class TestGenerateMinimalSnippet:
    """Tests for generate_minimal_snippet function."""

    def test_minimal_classification(self):
        """Test minimal snippet for classification."""
        snippet = generate_minimal_snippet(
            LogisticRegression,
            {"C": 0.5},
            task_type="classification"
        )

        assert "LogisticRegression" in snippet
        assert "C=0.5" in snippet
        assert "fit" in snippet
        assert "predict" in snippet

    def test_minimal_default_params(self):
        """Test minimal snippet with default params."""
        snippet = generate_minimal_snippet(
            LogisticRegression,
            {},
            task_type="classification"
        )

        assert "LogisticRegression()" in snippet

    def test_minimal_with_non_default(self):
        """Test minimal snippet filters default params."""
        snippet = generate_minimal_snippet(
            LogisticRegression,
            {"C": 1.0, "max_iter": 200},  # C=1.0 is default
            task_type="classification"
        )

        # C=1.0 should be filtered out as it's default
        # max_iter=200 should appear as it's non-default
        assert "max_iter=200" in snippet
