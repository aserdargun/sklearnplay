"""Code snippet generation for reproducibility.

Generates Python code that reproduces the current pipeline configuration.
"""

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def get_import_statement(class_obj: type) -> str:
    """Get import statement for a class.

    Args:
        class_obj: The class to import

    Returns:
        Import statement string
    """
    module = class_obj.__module__
    name = class_obj.__name__
    return f"from {module} import {name}"


def format_value(value: Any) -> str:
    """Format a value for code generation.

    Args:
        value: The value to format

    Returns:
        Python code string representation
    """
    if value is None:
        return "None"
    elif isinstance(value, str):
        return repr(value)
    elif isinstance(value, bool):
        return str(value)
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, (list, tuple)):
        formatted = [format_value(v) for v in value]
        if isinstance(value, tuple):
            return f"({', '.join(formatted)})"
        return f"[{', '.join(formatted)}]"
    elif isinstance(value, dict):
        items = [f"{repr(k)}: {format_value(v)}" for k, v in value.items()]
        return "{" + ", ".join(items) + "}"
    elif callable(value):
        return str(value.__name__)
    else:
        return repr(value)


def generate_estimator_code(
    estimator: BaseEstimator,
    var_name: str = "estimator",
) -> tuple[list[str], str]:
    """Generate code for an estimator.

    Args:
        estimator: The estimator instance
        var_name: Variable name to use

    Returns:
        Tuple of (import_statements, code_string)
    """
    imports = [get_import_statement(type(estimator))]
    params = estimator.get_params(deep=False)

    # Filter to non-default params
    default_estimator = type(estimator)()
    default_params = default_estimator.get_params(deep=False)

    non_default = {
        k: v for k, v in params.items() if k in default_params and v != default_params[k]
    }

    class_name = type(estimator).__name__

    if non_default:
        param_str = ", ".join(f"{k}={format_value(v)}" for k, v in non_default.items())
        code = f"{var_name} = {class_name}({param_str})"
    else:
        code = f"{var_name} = {class_name}()"

    return imports, code


def generate_pipeline_code(
    pipeline: Pipeline,
    var_name: str = "pipeline",
) -> tuple[list[str], str]:
    """Generate code for a pipeline.

    Args:
        pipeline: The pipeline instance
        var_name: Variable name to use

    Returns:
        Tuple of (import_statements, code_string)
    """
    imports = ["from sklearn.pipeline import Pipeline"]
    step_codes = []

    for step_name, step in pipeline.steps:
        if isinstance(step, ColumnTransformer):
            step_imports, step_code = generate_column_transformer_code(
                step, f"{step_name}_transformer"
            )
            imports.extend(step_imports)
            step_codes.append(step_code)
            step_codes.append(f"    ('{step_name}', {step_name}_transformer),")
        elif isinstance(step, Pipeline):
            step_imports, step_code = generate_pipeline_code(step, f"{step_name}_pipe")
            imports.extend(step_imports)
            step_codes.append(step_code)
            step_codes.append(f"    ('{step_name}', {step_name}_pipe),")
        elif step == "passthrough":
            step_codes.append(f"    ('{step_name}', 'passthrough'),")
        else:
            step_imports, step_code = generate_estimator_code(step, f"{step_name}_step")
            imports.extend(step_imports)
            step_codes.append(step_code)
            step_codes.append(f"    ('{step_name}', {step_name}_step),")

    code_lines = [
        *step_codes[:-1],  # All step definitions
        f"{var_name} = Pipeline([",
        *[s for s in step_codes if s.startswith("    (")],
        "])",
    ]

    # Deduplicate imports
    imports = list(dict.fromkeys(imports))

    return imports, "\n".join(code_lines)


def generate_column_transformer_code(
    ct: ColumnTransformer,
    var_name: str = "preprocessor",
) -> tuple[list[str], str]:
    """Generate code for a ColumnTransformer.

    Args:
        ct: The ColumnTransformer instance
        var_name: Variable name to use

    Returns:
        Tuple of (import_statements, code_string)
    """
    imports = ["from sklearn.compose import ColumnTransformer"]
    transformer_defs = []
    transformer_tuples = []

    for name, transformer, columns in ct.transformers_:
        if transformer == "drop":
            transformer_tuples.append(f"    ('{name}', 'drop', {format_value(columns)}),")
        elif transformer == "passthrough":
            transformer_tuples.append(f"    ('{name}', 'passthrough', {format_value(columns)}),")
        elif isinstance(transformer, Pipeline):
            t_imports, t_code = generate_pipeline_code(transformer, f"{name}_pipe")
            imports.extend(t_imports)
            transformer_defs.append(t_code)
            transformer_tuples.append(f"    ('{name}', {name}_pipe, {format_value(columns)}),")
        else:
            t_imports, t_code = generate_estimator_code(transformer, f"{name}_transformer")
            imports.extend(t_imports)
            transformer_defs.append(t_code)
            transformer_tuples.append(
                f"    ('{name}', {name}_transformer, {format_value(columns)}),"
            )

    code_lines = [
        *transformer_defs,
        f"{var_name} = ColumnTransformer([",
        *transformer_tuples,
        "])",
    ]

    imports = list(dict.fromkeys(imports))

    return imports, "\n".join(code_lines)


def generate_code_snippet(
    pipeline: Pipeline | BaseEstimator,
    X_train_shape: tuple[int, int],
    y_train_shape: tuple[int] | None,
    test_size: float = 0.2,
    random_state: int = 42,
    dataset_name: str = "your_dataset",
    task_type: str = "classification",
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> str:
    """Generate a complete reproducible code snippet.

    Args:
        pipeline: The fitted pipeline or estimator
        X_train_shape: Shape of training data
        y_train_shape: Shape of target
        test_size: Test set proportion
        random_state: Random seed
        dataset_name: Name of dataset for comments
        task_type: The task type
        numeric_cols: Numeric column names
        categorical_cols: Categorical column names

    Returns:
        Complete Python code string
    """
    # Generate pipeline code
    if isinstance(pipeline, Pipeline):
        imports, pipeline_code = generate_pipeline_code(pipeline)
    else:
        imports, pipeline_code = generate_estimator_code(pipeline, "model")
        pipeline_code = pipeline_code.replace("estimator", "model")

    # Add common imports
    common_imports = [
        "import numpy as np",
        "import pandas as pd",
        "from sklearn.model_selection import train_test_split",
    ]

    if task_type == "classification":
        common_imports.append("from sklearn.metrics import accuracy_score, classification_report")
    elif task_type == "regression":
        common_imports.append("from sklearn.metrics import r2_score, mean_squared_error")
    elif task_type == "clustering":
        common_imports.append("from sklearn.metrics import silhouette_score")

    all_imports = common_imports + imports
    all_imports = list(dict.fromkeys(all_imports))

    # Generate full snippet
    snippet_parts = [
        '"""',
        f"Reproducible ML Pipeline - {dataset_name}",
        f"Task: {task_type}",
        f"Training samples: {X_train_shape[0]}, Features: {X_train_shape[1]}",
        '"""',
        "",
        "# Imports",
        *all_imports,
        "",
        "# Load your data",
        "# Replace this with your actual data loading code",
        f"# X = pd.read_csv('{dataset_name}.csv')",
        "# y = X.pop('target_column')",
        "",
    ]

    if numeric_cols or categorical_cols:
        snippet_parts.extend(
            [
                "# Column definitions",
                f"numeric_cols = {format_value(numeric_cols or [])}",
                f"categorical_cols = {format_value(categorical_cols or [])}",
                "",
            ]
        )

    snippet_parts.extend(
        [
            "# Train/test split",
            "X_train, X_test, y_train, y_test = train_test_split(",
            f"    X, y, test_size={test_size}, random_state={random_state}",
            ")",
            "",
            "# Build pipeline",
            pipeline_code,
            "",
            "# Fit the model",
        ]
    )

    if isinstance(pipeline, Pipeline):
        snippet_parts.append("pipeline.fit(X_train, y_train)")
    else:
        snippet_parts.append("model.fit(X_train, y_train)")

    snippet_parts.append("")
    snippet_parts.append("# Predictions")

    if isinstance(pipeline, Pipeline):
        model_var = "pipeline"
    else:
        model_var = "model"

    snippet_parts.append(f"y_pred = {model_var}.predict(X_test)")

    # Add evaluation code
    snippet_parts.append("")
    snippet_parts.append("# Evaluation")

    if task_type == "classification":
        snippet_parts.extend(
            [
                "print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')",
                "print(classification_report(y_test, y_pred))",
            ]
        )
    elif task_type == "regression":
        snippet_parts.extend(
            [
                "print(f'R² Score: {r2_score(y_test, y_pred):.4f}')",
                "print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}')",
            ]
        )
    elif task_type == "clustering":
        snippet_parts.extend(
            [
                "labels = y_pred",
                "print(f'Silhouette Score: {silhouette_score(X_test, labels):.4f}')",
            ]
        )

    return "\n".join(snippet_parts)


def generate_minimal_snippet(
    estimator_class: type,
    estimator_params: dict,
    task_type: str = "classification",
) -> str:
    """Generate a minimal code snippet for quick reference.

    Args:
        estimator_class: The estimator class
        estimator_params: Parameters to use
        task_type: The task type

    Returns:
        Minimal Python code string
    """
    import_stmt = get_import_statement(estimator_class)
    class_name = estimator_class.__name__

    # Filter non-default params
    try:
        default_params = estimator_class().get_params(deep=False)
        non_default = {
            k: v
            for k, v in estimator_params.items()
            if k in default_params and v != default_params[k]
        }
    except Exception:
        non_default = estimator_params

    if non_default:
        param_str = ", ".join(f"{k}={format_value(v)}" for k, v in non_default.items())
        init_code = f"model = {class_name}({param_str})"
    else:
        init_code = f"model = {class_name}()"

    snippet = f"""{import_stmt}

{init_code}
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
"""

    return snippet
