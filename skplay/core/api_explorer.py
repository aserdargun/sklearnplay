"""API Explorer utilities using Python introspection.

Provides tools to explore scikit-learn's API structure, signatures, and documentation.
"""

import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from sklearn.utils import all_estimators

try:
    from sklearn.utils.discovery import all_displays

    DISCOVERY_AVAILABLE = True
except ImportError:
    DISCOVERY_AVAILABLE = False


@dataclass
class APIEntry:
    """Represents an API entry (class or function)."""

    name: str
    module: str
    kind: Literal["class", "function", "display"]
    signature: str
    short_doc: str
    full_doc: str
    parameters: list[dict] = field(default_factory=list)
    is_estimator: bool = False
    estimator_type: str | None = None
    tags: list[str] = field(default_factory=list)


def get_short_docstring(obj: Any, max_lines: int = 3) -> str:
    """Extract short docstring from an object.

    Args:
        obj: The object with a docstring
        max_lines: Maximum lines to include

    Returns:
        Short docstring
    """
    doc = inspect.getdoc(obj)
    if not doc:
        return ""

    lines = doc.split("\n")
    short_lines: list[str] = []

    for line in lines[:max_lines]:
        if line.strip() == "" and short_lines:
            break
        short_lines.append(line)

    return "\n".join(short_lines).strip()


def get_signature_string(obj: Any) -> str:
    """Get signature string for a callable.

    Args:
        obj: The callable object

    Returns:
        Signature string
    """
    try:
        sig = inspect.signature(obj)
        return str(sig)
    except (ValueError, TypeError):
        return "()"


def get_parameter_info(obj: Any) -> list[dict]:
    """Extract parameter information from signature and docstring.

    Args:
        obj: The callable object

    Returns:
        List of parameter dictionaries
    """
    params: list[dict[str, Any]] = []

    try:
        sig = inspect.signature(obj)
    except (ValueError, TypeError):
        return params

    doc = inspect.getdoc(obj) or ""

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        param_info = {
            "name": name,
            "kind": str(param.kind.name),
            "default": None if param.default is inspect.Parameter.empty else repr(param.default),
            "annotation": str(param.annotation)
            if param.annotation is not inspect.Parameter.empty
            else None,
            "description": "",
        }

        # Try to extract description from docstring
        # Look for "name : type" or "name: type" pattern
        pattern = rf"\b{re.escape(name)}\s*:\s*[^\n]+\n\s*(.+?)(?=\n\s*\w+\s*:|$)"
        match = re.search(pattern, doc, re.DOTALL)
        if match:
            param_info["description"] = match.group(1).strip()[:200]

        params.append(param_info)

    return params


def get_estimator_type(estimator_class: type) -> str | None:
    """Determine the type of an estimator.

    Args:
        estimator_class: The estimator class

    Returns:
        Type string or None
    """
    if hasattr(estimator_class, "_estimator_type"):
        return str(estimator_class._estimator_type)

    # Check by inheritance
    try:
        from sklearn.base import (
            ClassifierMixin,
            ClusterMixin,
            OutlierMixin,
            RegressorMixin,
            TransformerMixin,
        )

        if issubclass(estimator_class, ClassifierMixin):
            return "classifier"
        elif issubclass(estimator_class, RegressorMixin):
            return "regressor"
        elif issubclass(estimator_class, ClusterMixin):
            return "clusterer"
        elif issubclass(estimator_class, TransformerMixin):
            return "transformer"
        elif issubclass(estimator_class, OutlierMixin):
            return "outlier_detector"
    except Exception:
        pass

    return None


def get_tags_from_module(module_path: str) -> list[str]:
    """Extract tags from module path.

    Args:
        module_path: The module path

    Returns:
        List of tags
    """
    tags = []

    parts = module_path.split(".")

    # Common module -> tag mappings
    module_tags = {
        "linear_model": ["linear"],
        "tree": ["tree", "decision-tree"],
        "ensemble": ["ensemble"],
        "svm": ["svm", "kernel"],
        "neighbors": ["neighbors", "knn"],
        "naive_bayes": ["naive-bayes", "probabilistic"],
        "neural_network": ["neural-network", "mlp"],
        "cluster": ["clustering"],
        "decomposition": ["decomposition", "dimensionality-reduction"],
        "preprocessing": ["preprocessing", "transformation"],
        "feature_selection": ["feature-selection"],
        "feature_extraction": ["feature-extraction"],
        "model_selection": ["model-selection", "validation"],
        "metrics": ["metrics", "scoring"],
        "impute": ["imputation", "missing-values"],
        "compose": ["pipeline", "composition"],
        "calibration": ["calibration", "probability"],
        "manifold": ["manifold", "dimensionality-reduction"],
        "covariance": ["covariance", "outlier-detection"],
        "discriminant_analysis": ["discriminant-analysis"],
        "gaussian_process": ["gaussian-process", "probabilistic"],
        "semi_supervised": ["semi-supervised"],
        "cross_decomposition": ["cross-decomposition"],
        "isotonic": ["isotonic", "regression"],
        "kernel_ridge": ["kernel", "ridge"],
        "kernel_approximation": ["kernel", "approximation"],
        "mixture": ["mixture", "clustering", "probabilistic"],
    }

    for part in parts:
        if part in module_tags:
            tags.extend(module_tags[part])

    return list(set(tags))


def build_api_entry(
    name: str, obj: Any, module: str, kind: Literal["class", "function", "display"]
) -> APIEntry:
    """Build an APIEntry from an object.

    Args:
        name: Object name
        obj: The object
        module: Module path
        kind: Entry kind

    Returns:
        APIEntry instance
    """
    is_estimator = kind == "class" and hasattr(obj, "fit")
    estimator_type = get_estimator_type(obj) if is_estimator else None
    tags = get_tags_from_module(module)

    if estimator_type:
        tags.append(estimator_type)

    return APIEntry(
        name=name,
        module=module,
        kind=kind,
        signature=get_signature_string(obj),
        short_doc=get_short_docstring(obj),
        full_doc=inspect.getdoc(obj) or "",
        parameters=get_parameter_info(obj),
        is_estimator=is_estimator,
        estimator_type=estimator_type,
        tags=tags,
    )


class APIExplorer:
    """Explorer for scikit-learn API."""

    def __init__(self):
        """Initialize the API explorer."""
        self._entries: dict[str, APIEntry] = {}
        self._by_module: dict[str, list[str]] = {}
        self._by_kind: dict[str, list[str]] = {}
        self._by_type: dict[str, list[str]] = {}
        self._loaded = False

    def load(self) -> None:
        """Load all API entries from scikit-learn."""
        if self._loaded:
            return

        # Load all estimators
        for name, est_class in all_estimators():
            try:
                module = est_class.__module__
                entry = build_api_entry(name, est_class, module, "class")
                self._entries[name] = entry
                self._index_entry(entry)
            except Exception:
                continue

        # Load displays if available
        if DISCOVERY_AVAILABLE:
            try:
                for name, display_class in all_displays():
                    try:
                        module = display_class.__module__
                        entry = build_api_entry(name, display_class, module, "display")
                        self._entries[name] = entry
                        self._index_entry(entry)
                    except Exception:
                        continue
            except Exception:
                pass

        # Add common functions manually (discovery may not catch all)
        self._add_common_functions()

        self._loaded = True

    def _add_common_functions(self) -> None:
        """Add commonly used functions to the registry."""
        common_functions = [
            (
                "sklearn.model_selection",
                [
                    "train_test_split",
                    "cross_val_score",
                    "cross_validate",
                    "GridSearchCV",
                    "RandomizedSearchCV",
                ],
            ),
            (
                "sklearn.metrics",
                [
                    "accuracy_score",
                    "precision_score",
                    "recall_score",
                    "f1_score",
                    "roc_auc_score",
                    "confusion_matrix",
                    "classification_report",
                    "mean_squared_error",
                    "r2_score",
                    "mean_absolute_error",
                    "silhouette_score",
                ],
            ),
            (
                "sklearn.preprocessing",
                [
                    "StandardScaler",
                    "MinMaxScaler",
                    "LabelEncoder",
                    "OneHotEncoder",
                    "OrdinalEncoder",
                ],
            ),
            ("sklearn.pipeline", ["Pipeline", "make_pipeline"]),
            ("sklearn.compose", ["ColumnTransformer", "make_column_transformer"]),
        ]

        for module_path, names in common_functions:
            try:
                module = __import__(module_path, fromlist=names)
                for name in names:
                    if name in self._entries:
                        continue
                    try:
                        obj = getattr(module, name)
                        kind: Literal["class", "function", "display"] = (
                            "class" if inspect.isclass(obj) else "function"
                        )
                        entry = build_api_entry(name, obj, module_path, kind)
                        self._entries[name] = entry
                        self._index_entry(entry)
                    except Exception:
                        continue
            except Exception:
                continue

    def _index_entry(self, entry: APIEntry) -> None:
        """Index an entry for fast lookup.

        Args:
            entry: The entry to index
        """
        # By module
        if entry.module not in self._by_module:
            self._by_module[entry.module] = []
        self._by_module[entry.module].append(entry.name)

        # By kind
        if entry.kind not in self._by_kind:
            self._by_kind[entry.kind] = []
        self._by_kind[entry.kind].append(entry.name)

        # By estimator type
        if entry.estimator_type:
            if entry.estimator_type not in self._by_type:
                self._by_type[entry.estimator_type] = []
            self._by_type[entry.estimator_type].append(entry.name)

    def search(
        self,
        query: str,
        kind: str | None = None,
        estimator_type: str | None = None,
        limit: int = 50,
    ) -> list[APIEntry]:
        """Search for API entries.

        Args:
            query: Search query (matches name and docstring)
            kind: Filter by kind (class, function, display)
            estimator_type: Filter by estimator type
            limit: Maximum results

        Returns:
            List of matching entries
        """
        self.load()

        query_lower = query.lower()
        results = []

        for name, entry in self._entries.items():
            # Apply filters
            if kind and entry.kind != kind:
                continue
            if estimator_type and entry.estimator_type != estimator_type:
                continue

            # Score match
            score = 0
            name_lower = name.lower()

            if name_lower == query_lower:
                score = 100
            elif name_lower.startswith(query_lower):
                score = 80
            elif query_lower in name_lower:
                score = 60
            elif query_lower in entry.short_doc.lower():
                score = 40
            elif any(query_lower in tag for tag in entry.tags):
                score = 30

            if score > 0:
                results.append((score, entry))

        # Sort by score and return
        results.sort(key=lambda x: (-x[0], x[1].name))
        return [entry for _, entry in results[:limit]]

    def get(self, name: str) -> APIEntry | None:
        """Get an entry by name.

        Args:
            name: Entry name

        Returns:
            APIEntry or None
        """
        self.load()
        return self._entries.get(name)

    def list_by_module(self, module: str) -> list[APIEntry]:
        """List entries in a module.

        Args:
            module: Module path

        Returns:
            List of entries
        """
        self.load()
        names = self._by_module.get(module, [])
        return [self._entries[n] for n in names]

    def list_by_kind(self, kind: str) -> list[APIEntry]:
        """List entries of a kind.

        Args:
            kind: Entry kind

        Returns:
            List of entries
        """
        self.load()
        names = self._by_kind.get(kind, [])
        return [self._entries[n] for n in names]

    def list_by_type(self, estimator_type: str) -> list[APIEntry]:
        """List estimators of a type.

        Args:
            estimator_type: Estimator type

        Returns:
            List of entries
        """
        self.load()
        names = self._by_type.get(estimator_type, [])
        return [self._entries[n] for n in names]

    def get_modules(self) -> list[str]:
        """Get all indexed modules.

        Returns:
            List of module paths
        """
        self.load()
        return sorted(self._by_module.keys())

    def get_estimator_types(self) -> list[str]:
        """Get all estimator types.

        Returns:
            List of estimator types
        """
        self.load()
        return sorted(self._by_type.keys())

    def get_all_names(self) -> list[str]:
        """Get all entry names.

        Returns:
            Sorted list of names
        """
        self.load()
        return sorted(self._entries.keys())

    def get_module_hierarchy(self) -> dict[str, list[str]]:
        """Get module hierarchy for display.

        Returns:
            Dictionary of top-level module -> sub-modules
        """
        self.load()

        hierarchy: dict[str, list[str]] = {}

        for module in self._by_module.keys():
            parts = module.split(".")
            if len(parts) >= 2:
                top_level = parts[1] if parts[0] == "sklearn" else parts[0]
                if top_level not in hierarchy:
                    hierarchy[top_level] = []
                if module not in hierarchy[top_level]:
                    hierarchy[top_level].append(module)

        return hierarchy


# Global explorer instance
_explorer = None


def get_explorer() -> APIExplorer:
    """Get the global API explorer instance.

    Returns:
        APIExplorer instance
    """
    global _explorer
    if _explorer is None:
        _explorer = APIExplorer()
    return _explorer


def search_api(
    query: str,
    kind: str | None = None,
    estimator_type: str | None = None,
    limit: int = 50,
) -> list[APIEntry]:
    """Search the scikit-learn API.

    Args:
        query: Search query
        kind: Filter by kind
        estimator_type: Filter by estimator type
        limit: Maximum results

    Returns:
        List of matching entries
    """
    return get_explorer().search(query, kind, estimator_type, limit)


def get_api_entry(name: str) -> APIEntry | None:
    """Get an API entry by name.

    Args:
        name: Entry name

    Returns:
        APIEntry or None
    """
    return get_explorer().get(name)


def generate_example_snippet(entry: APIEntry) -> str:
    """Generate an example code snippet for an API entry.

    Args:
        entry: The API entry

    Returns:
        Example code string
    """
    if entry.kind == "function":
        return f"""from {entry.module} import {entry.name}

# {entry.short_doc[:80]}...
result = {entry.name}(...)
"""

    elif entry.is_estimator:
        return f"""from {entry.module} import {entry.name}

# {entry.short_doc[:80]}...
model = {entry.name}()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
"""

    else:
        return f"""from {entry.module} import {entry.name}

# {entry.short_doc[:80]}...
obj = {entry.name}(...)
"""
