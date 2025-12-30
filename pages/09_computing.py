"""Computing with scikit-learn.

This page covers performance, parallelism, and computational aspects.
"""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Computing", page_icon="images/icon.png", layout="wide")

from skplay.ui.level import get_level, level_selector


def main():
    st.title("⚡ Computing with scikit-learn")

    with st.sidebar:
        level_selector()

    st.markdown("""
    Optimize performance and scale to larger datasets.

    [📚 sklearn User Guide: Computing](https://scikit-learn.org/stable/computing.html)
    """)

    topic = st.radio(
        "Topic",
        options=["parallelism", "memory", "performance", "strategies"],
        format_func=lambda x: x.title(),
        horizontal=True,
    )

    st.markdown("---")

    if topic == "parallelism":
        parallelism_section()
    elif topic == "memory":
        memory_section()
    elif topic == "performance":
        performance_section()
    else:
        strategies_section()


def parallelism_section():
    """Parallelism in sklearn."""
    st.header("Parallelism")

    st.markdown("""
    Many sklearn estimators support parallel execution via the `n_jobs` parameter.
    """)

    st.subheader("Using n_jobs")

    st.code(
        """
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Use all CPU cores
model = RandomForestClassifier(n_jobs=-1)

# Or specify number of jobs
model = RandomForestClassifier(n_jobs=4)

# Cross-validation in parallel
scores = cross_val_score(model, X, y, cv=5, n_jobs=-1)
    """,
        language="python",
    )

    st.subheader("Common n_jobs Values")

    njobs_data = [
        {"Value": "1", "Meaning": "Sequential execution (no parallelism)"},
        {"Value": "-1", "Meaning": "Use all available CPU cores"},
        {"Value": "-2", "Meaning": "Use all cores except one"},
        {"Value": "n", "Meaning": "Use exactly n cores"},
    ]

    st.dataframe(pd.DataFrame(njobs_data), hide_index=True, width="stretch")

    st.subheader("Backend Configuration")

    level = get_level()

    if level in ("intermediate", "advanced"):
        st.code(
            """
import joblib

# Use threading backend (for I/O bound tasks)
with joblib.parallel_backend('threading'):
    model.fit(X, y)

# Use loky backend (default, for CPU bound tasks)
with joblib.parallel_backend('loky'):
    model.fit(X, y)
        """,
            language="python",
        )

    st.warning("""
    **Note:** Parallelism has overhead. For small datasets, sequential may be faster.
    Also be careful with nested parallelism (e.g., parallel cross-validation with parallel estimators).
    """)


def memory_section():
    """Memory efficiency."""
    st.header("Memory Efficiency")

    st.markdown("""
    Tips for handling large datasets with limited memory.
    """)

    st.subheader("Sparse Matrices")

    st.code(
        """
from scipy.sparse import csr_matrix

# Many sklearn estimators work with sparse matrices
X_sparse = csr_matrix(X)  # Convert dense to sparse

# Check if estimator supports sparse
from sklearn.utils.estimator_checks import check_estimator
# Or check documentation
    """,
        language="python",
    )

    st.subheader("Incremental Learning")

    st.markdown("""
    Some estimators support `partial_fit` for streaming/batch learning:
    """)

    st.code(
        """
from sklearn.linear_model import SGDClassifier

model = SGDClassifier()

# Fit in batches
for X_batch, y_batch in data_generator:
    model.partial_fit(X_batch, y_batch, classes=all_classes)
    """,
        language="python",
    )

    st.subheader("Estimators with partial_fit")

    estimators_data = [
        {
            "Category": "Classification",
            "Estimators": "SGDClassifier, MultinomialNB, BernoulliNB, Perceptron",
        },
        {"Category": "Regression", "Estimators": "SGDRegressor"},
        {"Category": "Clustering", "Estimators": "MiniBatchKMeans, Birch"},
        {"Category": "Decomposition", "Estimators": "IncrementalPCA, MiniBatchDictionaryLearning"},
    ]

    st.dataframe(pd.DataFrame(estimators_data), hide_index=True, width="stretch")


def performance_section():
    """Performance tips."""
    st.header("Performance Tips")

    tips = [
        (
            "Use appropriate data types",
            "float32 instead of float64 can halve memory and improve speed",
        ),
        ("Scale features", "Many algorithms converge faster with scaled data"),
        ("Use sparse matrices", "For high-dimensional sparse data (e.g., text)"),
        ("Enable n_jobs", "Use all cores for parallelizable operations"),
        ("Use appropriate algorithms", "e.g., LinearSVC is faster than SVC with linear kernel"),
        ("Reduce data dimensionally", "PCA or feature selection before complex algorithms"),
        ("Use approximate algorithms", "e.g., MiniBatchKMeans instead of KMeans"),
        ("Cache transformations", "Use joblib.Memory or sklearn's memory parameter"),
    ]

    for tip, desc in tips:
        st.markdown(f"**{tip}:** {desc}")

    st.subheader("Algorithm Complexity")

    complexity_data = [
        {"Algorithm": "Linear Models", "Training": "O(n × d)", "Prediction": "O(d)"},
        {"Algorithm": "Decision Trees", "Training": "O(n × d × log n)", "Prediction": "O(log n)"},
        {
            "Algorithm": "Random Forest",
            "Training": "O(k × n × d × log n)",
            "Prediction": "O(k × log n)",
        },
        {"Algorithm": "KNN", "Training": "O(1) or O(n)", "Prediction": "O(n × d)"},
        {"Algorithm": "SVM (RBF)", "Training": "O(n² ~ n³)", "Prediction": "O(n_sv × d)"},
        {"Algorithm": "K-Means", "Training": "O(n × k × d × i)", "Prediction": "O(k × d)"},
    ]

    st.dataframe(pd.DataFrame(complexity_data), hide_index=True, width="stretch")
    st.caption("n=samples, d=features, k=clusters/trees, i=iterations, n_sv=support vectors")


def strategies_section():
    """Strategies for large data."""
    st.header("Strategies for Large Datasets")

    level = get_level()

    st.subheader("When Data Fits in Memory")

    st.markdown("""
    1. **Subsample for exploration**: Use a random sample to find good hyperparameters
    2. **Use efficient algorithms**: LinearSVC, SGDClassifier, HistGradientBoosting
    3. **Enable parallelism**: Set n_jobs=-1
    4. **Reduce dimensionality first**: PCA, feature selection
    """)

    st.subheader("When Data Doesn't Fit in Memory")

    st.markdown("""
    1. **Incremental learning**: Use partial_fit with data batches
    2. **Out-of-core learning**: Stream data from disk
    3. **Use memory-mapped arrays**: numpy memmap
    4. **Sample-based algorithms**: SGD, MiniBatch variants
    """)

    if level == "advanced":
        st.subheader("Distributed Computing")

        st.markdown("""
        For truly large scale, consider:

        - **Dask-ML**: Distributed sklearn-like API
        - **Spark MLlib**: Distributed ML on Spark
        - **Ray**: Distributed hyperparameter tuning

        ```python
        # Example with Dask
        from dask_ml.model_selection import GridSearchCV
        from dask.distributed import Client

        client = Client()  # Start Dask cluster
        search = GridSearchCV(model, param_grid)
        search.fit(X, y)
        ```
        """)


if __name__ == "__main__":
    main()
