"""Core playground engine components."""

from skplay.core.datasets import DatasetRegistry, get_dataset, list_datasets
from skplay.core.estimators import EstimatorRegistry, get_estimators_for_task
from skplay.core.evaluation import evaluate_model, get_metrics_for_task
from skplay.core.preprocessing import PreprocessingBuilder
from skplay.core.snippets import generate_code_snippet

__all__ = [
    "DatasetRegistry",
    "get_dataset",
    "list_datasets",
    "PreprocessingBuilder",
    "EstimatorRegistry",
    "get_estimators_for_task",
    "evaluate_model",
    "get_metrics_for_task",
    "generate_code_snippet",
]
