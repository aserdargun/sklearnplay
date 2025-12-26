"""Estimator registry and task-based selection.

Maps task types to available estimators with level-based filtering.
"""

from typing import Literal, Any
from dataclasses import dataclass, field
from sklearn.base import BaseEstimator

# Classification
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Regression
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    SGDRegressor,
    BayesianRidge,
    HuberRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Clustering
from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    AgglomerativeClustering,
    DBSCAN,
    OPTICS,
    SpectralClustering,
    Birch,
    MeanShift,
)
from sklearn.mixture import GaussianMixture

# Outlier detection
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

from skplay.core.datasets import TaskType

Level = Literal["beginner", "intermediate", "advanced"]


@dataclass
class EstimatorInfo:
    """Information about an estimator."""
    name: str
    class_: type
    description: str
    level: Level
    key_params: list[str] = field(default_factory=list)
    tips: str = ""
    doc_url: str = ""


class EstimatorRegistry:
    """Registry of available estimators by task type."""

    _estimators: dict[TaskType, list[EstimatorInfo]] = {
        "classification": [],
        "regression": [],
        "clustering": [],
        "outlier_detection": [],
    }

    @classmethod
    def register(cls, task_type: TaskType, info: EstimatorInfo) -> None:
        """Register an estimator for a task type."""
        cls._estimators[task_type].append(info)

    @classmethod
    def get_for_task(
        cls,
        task_type: TaskType,
        level: Level = "advanced",
    ) -> list[EstimatorInfo]:
        """Get estimators for a task type, filtered by level."""
        level_order = {"beginner": 0, "intermediate": 1, "advanced": 2}
        max_level = level_order[level]

        return [
            info for info in cls._estimators[task_type]
            if level_order[info.level] <= max_level
        ]

    @classmethod
    def get_by_name(cls, task_type: TaskType, name: str) -> EstimatorInfo | None:
        """Get a specific estimator by name."""
        for info in cls._estimators[task_type]:
            if info.name == name:
                return info
        return None

    @classmethod
    def list_names(cls, task_type: TaskType, level: Level = "advanced") -> list[str]:
        """List estimator names for a task type."""
        return [info.name for info in cls.get_for_task(task_type, level)]


# =============================================================================
# Register Classification Estimators
# =============================================================================

CLASSIFICATION_ESTIMATORS = [
    EstimatorInfo(
        name="Logistic Regression",
        class_=LogisticRegression,
        description="Linear model for classification. Fast and interpretable.",
        level="beginner",
        key_params=["C", "penalty", "solver", "max_iter"],
        tips="Start here for binary/multiclass problems. Increase max_iter if not converging.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
    ),
    EstimatorInfo(
        name="Decision Tree",
        class_=DecisionTreeClassifier,
        description="Tree-based model. Easy to interpret but prone to overfitting.",
        level="beginner",
        key_params=["max_depth", "min_samples_split", "min_samples_leaf", "criterion"],
        tips="Limit max_depth to prevent overfitting. Great for understanding feature importance.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html",
    ),
    EstimatorInfo(
        name="Random Forest",
        class_=RandomForestClassifier,
        description="Ensemble of decision trees. Robust and accurate.",
        level="beginner",
        key_params=["n_estimators", "max_depth", "min_samples_split", "max_features"],
        tips="More trees = better (but slower). Good default choice for most problems.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
    ),
    EstimatorInfo(
        name="K-Nearest Neighbors",
        class_=KNeighborsClassifier,
        description="Classify based on nearest training examples.",
        level="beginner",
        key_params=["n_neighbors", "weights", "metric"],
        tips="Scale features first. Use odd k for binary classification.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html",
    ),
    EstimatorInfo(
        name="Gaussian Naive Bayes",
        class_=GaussianNB,
        description="Probabilistic classifier assuming feature independence.",
        level="beginner",
        key_params=["var_smoothing"],
        tips="Very fast and works well with high-dimensional data. Assumes gaussian features.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html",
    ),
    EstimatorInfo(
        name="SVM (RBF Kernel)",
        class_=SVC,
        description="Support Vector Machine with RBF kernel. Powerful for complex boundaries.",
        level="intermediate",
        key_params=["C", "gamma", "kernel"],
        tips="Scale features! C controls regularization, gamma controls kernel width.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html",
    ),
    EstimatorInfo(
        name="Linear SVM",
        class_=LinearSVC,
        description="Linear Support Vector Machine. Fast for large datasets.",
        level="intermediate",
        key_params=["C", "loss", "max_iter"],
        tips="Faster than SVC with linear kernel. Good for text classification.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html",
    ),
    EstimatorInfo(
        name="Gradient Boosting",
        class_=GradientBoostingClassifier,
        description="Sequential ensemble that corrects previous errors.",
        level="intermediate",
        key_params=["n_estimators", "learning_rate", "max_depth", "subsample"],
        tips="Lower learning_rate + more trees = better but slower.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html",
    ),
    EstimatorInfo(
        name="Hist Gradient Boosting",
        class_=HistGradientBoostingClassifier,
        description="Fast gradient boosting with native missing value support.",
        level="intermediate",
        key_params=["learning_rate", "max_iter", "max_depth", "l2_regularization"],
        tips="Faster than GradientBoosting for larger datasets. Handles missing values.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html",
    ),
    EstimatorInfo(
        name="AdaBoost",
        class_=AdaBoostClassifier,
        description="Boosting with adaptive sample weights.",
        level="intermediate",
        key_params=["n_estimators", "learning_rate", "algorithm"],
        tips="Less prone to overfitting than other boosting methods.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html",
    ),
    EstimatorInfo(
        name="Extra Trees",
        class_=ExtraTreesClassifier,
        description="Extremely randomized trees. Faster than Random Forest.",
        level="intermediate",
        key_params=["n_estimators", "max_depth", "min_samples_split"],
        tips="More randomness than RF, sometimes better generalization.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html",
    ),
    EstimatorInfo(
        name="Ridge Classifier",
        class_=RidgeClassifier,
        description="Linear classifier with L2 regularization.",
        level="intermediate",
        key_params=["alpha", "solver"],
        tips="Fast linear model, good when classes are well-separated.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html",
    ),
    EstimatorInfo(
        name="SGD Classifier",
        class_=SGDClassifier,
        description="Linear model trained with stochastic gradient descent.",
        level="advanced",
        key_params=["loss", "penalty", "alpha", "learning_rate", "max_iter"],
        tips="Scales well to large datasets. Many loss functions available.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html",
    ),
    EstimatorInfo(
        name="MLP Classifier",
        class_=MLPClassifier,
        description="Multi-layer perceptron neural network.",
        level="advanced",
        key_params=["hidden_layer_sizes", "activation", "solver", "alpha", "learning_rate"],
        tips="Scale features! May need tuning of architecture and learning rate.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html",
    ),
    EstimatorInfo(
        name="LDA",
        class_=LinearDiscriminantAnalysis,
        description="Linear Discriminant Analysis. Also useful for dim reduction.",
        level="advanced",
        key_params=["solver", "shrinkage", "n_components"],
        tips="Works well when classes have similar covariance.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html",
    ),
    EstimatorInfo(
        name="QDA",
        class_=QuadraticDiscriminantAnalysis,
        description="Quadratic Discriminant Analysis. Allows class-specific covariance.",
        level="advanced",
        key_params=["reg_param"],
        tips="More flexible than LDA but needs more data.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html",
    ),
    EstimatorInfo(
        name="Bagging Classifier",
        class_=BaggingClassifier,
        description="Bootstrap aggregating with any base estimator.",
        level="advanced",
        key_params=["n_estimators", "max_samples", "max_features", "bootstrap"],
        tips="Reduces variance. Useful with unstable estimators.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html",
    ),
]

for info in CLASSIFICATION_ESTIMATORS:
    EstimatorRegistry.register("classification", info)


# =============================================================================
# Register Regression Estimators
# =============================================================================

REGRESSION_ESTIMATORS = [
    EstimatorInfo(
        name="Linear Regression",
        class_=LinearRegression,
        description="Ordinary least squares regression.",
        level="beginner",
        key_params=["fit_intercept"],
        tips="Start here. Simple and interpretable baseline.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html",
    ),
    EstimatorInfo(
        name="Ridge Regression",
        class_=Ridge,
        description="Linear regression with L2 regularization.",
        level="beginner",
        key_params=["alpha", "solver"],
        tips="Use when features are correlated. alpha controls regularization.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html",
    ),
    EstimatorInfo(
        name="Decision Tree Regressor",
        class_=DecisionTreeRegressor,
        description="Tree-based regressor. Easy to interpret.",
        level="beginner",
        key_params=["max_depth", "min_samples_split", "min_samples_leaf"],
        tips="Limit depth to prevent overfitting.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html",
    ),
    EstimatorInfo(
        name="Random Forest Regressor",
        class_=RandomForestRegressor,
        description="Ensemble of decision trees for regression.",
        level="beginner",
        key_params=["n_estimators", "max_depth", "min_samples_split"],
        tips="Robust default choice. More trees generally better.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html",
    ),
    EstimatorInfo(
        name="K-Nearest Neighbors Regressor",
        class_=KNeighborsRegressor,
        description="Predict based on average of nearest neighbors.",
        level="beginner",
        key_params=["n_neighbors", "weights", "metric"],
        tips="Scale features first. Choose k based on CV.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html",
    ),
    EstimatorInfo(
        name="Lasso Regression",
        class_=Lasso,
        description="Linear regression with L1 regularization (sparse coefficients).",
        level="intermediate",
        key_params=["alpha", "max_iter"],
        tips="Performs feature selection. Increase max_iter if not converging.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html",
    ),
    EstimatorInfo(
        name="ElasticNet",
        class_=ElasticNet,
        description="Linear regression with combined L1 and L2 regularization.",
        level="intermediate",
        key_params=["alpha", "l1_ratio", "max_iter"],
        tips="l1_ratio=0 is Ridge, l1_ratio=1 is Lasso.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html",
    ),
    EstimatorInfo(
        name="SVR (RBF Kernel)",
        class_=SVR,
        description="Support Vector Regression with RBF kernel.",
        level="intermediate",
        key_params=["C", "epsilon", "gamma"],
        tips="Scale features! epsilon defines insensitive tube width.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html",
    ),
    EstimatorInfo(
        name="Gradient Boosting Regressor",
        class_=GradientBoostingRegressor,
        description="Sequential ensemble for regression.",
        level="intermediate",
        key_params=["n_estimators", "learning_rate", "max_depth", "subsample"],
        tips="Lower learning_rate + more trees = better but slower.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html",
    ),
    EstimatorInfo(
        name="Hist Gradient Boosting Regressor",
        class_=HistGradientBoostingRegressor,
        description="Fast gradient boosting with native missing value support.",
        level="intermediate",
        key_params=["learning_rate", "max_iter", "max_depth", "l2_regularization"],
        tips="Faster than GradientBoosting for larger datasets.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html",
    ),
    EstimatorInfo(
        name="Extra Trees Regressor",
        class_=ExtraTreesRegressor,
        description="Extremely randomized trees for regression.",
        level="intermediate",
        key_params=["n_estimators", "max_depth", "min_samples_split"],
        tips="More randomness than RF. Often faster training.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html",
    ),
    EstimatorInfo(
        name="AdaBoost Regressor",
        class_=AdaBoostRegressor,
        description="Boosting regressor with adaptive sample weights.",
        level="intermediate",
        key_params=["n_estimators", "learning_rate", "loss"],
        tips="Less prone to overfitting. Try different loss functions.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html",
    ),
    EstimatorInfo(
        name="Huber Regressor",
        class_=HuberRegressor,
        description="Linear regression robust to outliers.",
        level="advanced",
        key_params=["epsilon", "alpha", "max_iter"],
        tips="Use when data has outliers. epsilon controls outlier threshold.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html",
    ),
    EstimatorInfo(
        name="Bayesian Ridge",
        class_=BayesianRidge,
        description="Bayesian linear regression with automatic regularization.",
        level="advanced",
        key_params=["alpha_1", "alpha_2", "lambda_1", "lambda_2"],
        tips="Automatically determines regularization. Provides uncertainty estimates.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html",
    ),
    EstimatorInfo(
        name="SGD Regressor",
        class_=SGDRegressor,
        description="Linear regression trained with stochastic gradient descent.",
        level="advanced",
        key_params=["loss", "penalty", "alpha", "learning_rate", "max_iter"],
        tips="Scales to large datasets. Many loss options.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html",
    ),
    EstimatorInfo(
        name="MLP Regressor",
        class_=MLPRegressor,
        description="Multi-layer perceptron for regression.",
        level="advanced",
        key_params=["hidden_layer_sizes", "activation", "solver", "alpha"],
        tips="Scale features! Tune architecture carefully.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html",
    ),
    EstimatorInfo(
        name="Linear SVR",
        class_=LinearSVR,
        description="Linear Support Vector Regression.",
        level="advanced",
        key_params=["C", "epsilon", "loss", "max_iter"],
        tips="Faster than SVR for linear problems.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html",
    ),
]

for info in REGRESSION_ESTIMATORS:
    EstimatorRegistry.register("regression", info)


# =============================================================================
# Register Clustering Estimators
# =============================================================================

CLUSTERING_ESTIMATORS = [
    EstimatorInfo(
        name="K-Means",
        class_=KMeans,
        description="Partition data into K clusters based on centroids.",
        level="beginner",
        key_params=["n_clusters", "init", "n_init", "max_iter"],
        tips="Scale features! Use elbow method to choose n_clusters.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html",
    ),
    EstimatorInfo(
        name="Agglomerative Clustering",
        class_=AgglomerativeClustering,
        description="Hierarchical clustering using bottom-up approach.",
        level="beginner",
        key_params=["n_clusters", "linkage", "metric"],
        tips="Good for hierarchical relationships. Use dendrogram to choose n_clusters.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html",
    ),
    EstimatorInfo(
        name="DBSCAN",
        class_=DBSCAN,
        description="Density-based clustering. Finds arbitrary-shaped clusters.",
        level="intermediate",
        key_params=["eps", "min_samples", "metric"],
        tips="No need to specify n_clusters. Tune eps using k-distance plot.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html",
    ),
    EstimatorInfo(
        name="Gaussian Mixture",
        class_=GaussianMixture,
        description="Soft clustering assuming gaussian component distributions.",
        level="intermediate",
        key_params=["n_components", "covariance_type", "max_iter"],
        tips="Provides probability of cluster membership. Use BIC for model selection.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html",
    ),
    EstimatorInfo(
        name="Mini-Batch K-Means",
        class_=MiniBatchKMeans,
        description="Faster variant of K-Means using mini-batches.",
        level="intermediate",
        key_params=["n_clusters", "batch_size", "max_iter"],
        tips="Use for larger datasets. Slightly less accurate than K-Means.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html",
    ),
    EstimatorInfo(
        name="OPTICS",
        class_=OPTICS,
        description="Ordering Points To Identify Clustering Structure.",
        level="advanced",
        key_params=["min_samples", "max_eps", "cluster_method", "xi"],
        tips="Similar to DBSCAN but handles varying densities.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html",
    ),
    EstimatorInfo(
        name="Spectral Clustering",
        class_=SpectralClustering,
        description="Clustering using graph Laplacian eigenvectors.",
        level="advanced",
        key_params=["n_clusters", "affinity", "n_neighbors"],
        tips="Good for non-convex clusters. Memory-intensive for large datasets.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html",
    ),
    EstimatorInfo(
        name="BIRCH",
        class_=Birch,
        description="Balanced Iterative Reducing and Clustering using Hierarchies.",
        level="advanced",
        key_params=["n_clusters", "threshold", "branching_factor"],
        tips="Efficient for large datasets. Two-phase: builds tree then clusters.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html",
    ),
    EstimatorInfo(
        name="Mean Shift",
        class_=MeanShift,
        description="Mode-finding clustering. Automatically determines n_clusters.",
        level="advanced",
        key_params=["bandwidth", "bin_seeding"],
        tips="No n_clusters needed. Use estimate_bandwidth() for bandwidth.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html",
    ),
]

for info in CLUSTERING_ESTIMATORS:
    EstimatorRegistry.register("clustering", info)


# =============================================================================
# Register Outlier Detection Estimators
# =============================================================================

OUTLIER_ESTIMATORS = [
    EstimatorInfo(
        name="Isolation Forest",
        class_=IsolationForest,
        description="Detect anomalies using isolation in random forests.",
        level="beginner",
        key_params=["n_estimators", "contamination", "max_samples", "max_features"],
        tips="Works well for high-dimensional data. contamination sets expected outlier fraction.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html",
    ),
    EstimatorInfo(
        name="Local Outlier Factor",
        class_=LocalOutlierFactor,
        description="Detect anomalies based on local density deviation.",
        level="intermediate",
        key_params=["n_neighbors", "contamination", "metric"],
        tips="Good for detecting local outliers. Set novelty=True for new data.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html",
    ),
    EstimatorInfo(
        name="One-Class SVM",
        class_=OneClassSVM,
        description="SVM-based novelty detection.",
        level="intermediate",
        key_params=["kernel", "nu", "gamma"],
        tips="nu is upper bound on outlier fraction. Scale features!",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html",
    ),
    EstimatorInfo(
        name="Elliptic Envelope",
        class_=EllipticEnvelope,
        description="Detect outliers assuming gaussian distribution.",
        level="advanced",
        key_params=["contamination", "support_fraction"],
        tips="Works best when data is approximately gaussian.",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html",
    ),
]

for info in OUTLIER_ESTIMATORS:
    EstimatorRegistry.register("outlier_detection", info)


# =============================================================================
# Utility functions
# =============================================================================

def get_estimators_for_task(
    task_type: TaskType,
    level: Level = "advanced",
) -> list[EstimatorInfo]:
    """Get available estimators for a task type and user level."""
    return EstimatorRegistry.get_for_task(task_type, level)


def get_estimator_names(task_type: TaskType, level: Level = "advanced") -> list[str]:
    """Get estimator names for a task type."""
    return EstimatorRegistry.list_names(task_type, level)


def get_estimator_class(task_type: TaskType, name: str) -> type | None:
    """Get the estimator class by name."""
    info = EstimatorRegistry.get_by_name(task_type, name)
    return info.class_ if info else None


def create_estimator(
    task_type: TaskType,
    name: str,
    **kwargs: Any,
) -> BaseEstimator:
    """Create an estimator instance with given parameters."""
    info = EstimatorRegistry.get_by_name(task_type, name)
    if not info:
        raise ValueError(f"Estimator '{name}' not found for task '{task_type}'")
    return info.class_(**kwargs)


def get_default_estimator(task_type: TaskType) -> str:
    """Get the default estimator name for a task type."""
    defaults = {
        "classification": "Random Forest",
        "regression": "Random Forest Regressor",
        "clustering": "K-Means",
        "outlier_detection": "Isolation Forest",
    }
    return defaults.get(task_type, "")
