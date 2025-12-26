"""Dataset Loading Utilities.

This page covers sklearn's dataset loading functions.
"""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dataset Loading", page_icon="📁", layout="wide")

from skplay.core.datasets import get_dataset
from skplay.ui.level import level_selector


def main():
    st.title("📁 Dataset Loading Utilities")

    with st.sidebar:
        level_selector()

    st.markdown("""
    sklearn provides utilities for loading datasets: built-in toy datasets,
    real-world datasets, and sample generators.

    [📚 sklearn User Guide: Dataset Loading](https://scikit-learn.org/stable/datasets.html)
    """)

    topic = st.radio(
        "Topic",
        options=["toy_datasets", "real_world", "generators", "external"],
        format_func={
            "toy_datasets": "Toy Datasets",
            "real_world": "Real-World Datasets",
            "generators": "Sample Generators",
            "external": "External Datasets",
        }.get,
        horizontal=True,
    )

    st.markdown("---")

    if topic == "toy_datasets":
        toy_datasets_section()
    elif topic == "real_world":
        real_world_section()
    elif topic == "generators":
        generators_section()
    else:
        external_section()


def toy_datasets_section():
    """Built-in toy datasets."""
    st.header("Toy Datasets")

    st.markdown("""
    Small datasets bundled with sklearn for quick experimentation.
    """)

    datasets = [
        {
            "Name": "load_iris",
            "Task": "Classification",
            "Samples": 150,
            "Features": 4,
            "Description": "Flower species classification",
        },
        {
            "Name": "load_digits",
            "Task": "Classification",
            "Samples": 1797,
            "Features": 64,
            "Description": "Handwritten digit recognition",
        },
        {
            "Name": "load_wine",
            "Task": "Classification",
            "Samples": 178,
            "Features": 13,
            "Description": "Wine cultivar classification",
        },
        {
            "Name": "load_breast_cancer",
            "Task": "Classification",
            "Samples": 569,
            "Features": 30,
            "Description": "Tumor malignancy classification",
        },
        {
            "Name": "load_diabetes",
            "Task": "Regression",
            "Samples": 442,
            "Features": 10,
            "Description": "Disease progression prediction",
        },
        {
            "Name": "load_linnerud",
            "Task": "Multioutput",
            "Samples": 20,
            "Features": 3,
            "Description": "Physical exercise dataset",
        },
    ]

    st.dataframe(pd.DataFrame(datasets), hide_index=True, use_container_width=True)

    st.code(
        """
from sklearn.datasets import load_iris, load_diabetes

# Load as Bunch object
iris = load_iris()
X, y = iris.data, iris.target
print(iris.feature_names)
print(iris.target_names)

# Load as DataFrame
iris = load_iris(as_frame=True)
X = iris.data  # pandas DataFrame
y = iris.target  # pandas Series
    """,
        language="python",
    )

    # Interactive explorer
    st.subheader("Explore Datasets")

    selected = st.selectbox(
        "Select dataset",
        options=["iris", "wine", "breast_cancer", "diabetes", "digits"],
    )

    if st.button("Load Dataset"):
        data = get_dataset(selected)

        st.markdown(f"**{data.card.name}**: {data.card.description}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Samples", data.card.n_samples)
        col2.metric("Features", data.card.n_features)
        col3.metric("Task", data.card.task_type)

        st.dataframe(data.X.head(10), use_container_width=True)


def real_world_section():
    """Real-world datasets."""
    st.header("Real-World Datasets")

    st.markdown("""
    Larger datasets downloaded on first use. Cached locally.
    """)

    datasets = [
        {
            "Name": "fetch_california_housing",
            "Task": "Regression",
            "Samples": "20,640",
            "Description": "California house prices",
        },
        {
            "Name": "fetch_covtype",
            "Task": "Classification",
            "Samples": "581,012",
            "Description": "Forest cover type",
        },
        {
            "Name": "fetch_kddcup99",
            "Task": "Classification",
            "Samples": "4.9M",
            "Description": "Network intrusion detection",
        },
        {
            "Name": "fetch_20newsgroups",
            "Task": "Text Classification",
            "Samples": "~18,000",
            "Description": "Newsgroup documents",
        },
        {
            "Name": "fetch_lfw_people",
            "Task": "Face Recognition",
            "Samples": "13,233",
            "Description": "Labeled faces in the wild",
        },
        {
            "Name": "fetch_olivetti_faces",
            "Task": "Face Recognition",
            "Samples": "400",
            "Description": "Olivetti faces",
        },
    ]

    st.dataframe(pd.DataFrame(datasets), hide_index=True, use_container_width=True)

    st.code(
        """
from sklearn.datasets import fetch_california_housing, fetch_20newsgroups

# Housing dataset
housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target

# Text dataset (specify subset)
newsgroups = fetch_20newsgroups(
    subset='train',
    categories=['sci.med', 'sci.space'],
    remove=('headers', 'footers', 'quotes'),
)
    """,
        language="python",
    )


def generators_section():
    """Sample generators."""
    st.header("Sample Generators")

    st.markdown("""
    Create synthetic datasets with controlled properties for testing.
    """)

    st.subheader("Classification")

    st.code(
        """
from sklearn.datasets import make_classification, make_blobs, make_moons

# General classification
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=3,
    random_state=42
)

# Gaussian blobs (for clustering too)
X, y = make_blobs(n_samples=500, centers=4, random_state=42)

# Moon shapes (nonlinear)
X, y = make_moons(n_samples=500, noise=0.1, random_state=42)
    """,
        language="python",
    )

    st.subheader("Regression")

    st.code(
        """
from sklearn.datasets import make_regression, make_friedman1

# Linear regression
X, y = make_regression(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    noise=10,
    random_state=42
)

# Nonlinear (Friedman #1)
X, y = make_friedman1(n_samples=1000, n_features=10, noise=0.1)
    """,
        language="python",
    )

    # Interactive demo
    st.subheader("Interactive Demo")

    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs, make_circles, make_classification, make_moons

    generator = st.selectbox(
        "Generator",
        ["make_blobs", "make_moons", "make_circles", "make_classification"],
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        n_samples = st.slider("Samples", 100, 1000, 500)
        if generator == "make_blobs":
            n_centers = st.slider("Centers", 2, 6, 3)
        elif generator in ["make_moons", "make_circles"]:
            noise = st.slider("Noise", 0.0, 0.5, 0.1)

    with col2:
        if generator == "make_blobs":
            X, y = make_blobs(n_samples=n_samples, centers=n_centers, random_state=42)
        elif generator == "make_moons":
            X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        elif generator == "make_circles":
            X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
        else:
            X, y = make_classification(
                n_samples=n_samples,
                n_features=2,
                n_informative=2,
                n_redundant=0,
                n_clusters_per_class=1,
                random_state=42,
            )

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", alpha=0.6, edgecolors="k")
        ax.set_title(f"{generator}(n_samples={n_samples})")
        plt.colorbar(scatter, ax=ax, label="Class")
        st.pyplot(fig)


def external_section():
    """Loading external data."""
    st.header("Loading External Data")

    st.markdown("""
    Load your own datasets from files or URLs.
    """)

    st.subheader("From Files")

    st.code(
        """
import pandas as pd

# CSV
df = pd.read_csv("data.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Excel
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")

# JSON
df = pd.read_json("data.json")

# Parquet (efficient for large data)
df = pd.read_parquet("data.parquet")
    """,
        language="python",
    )

    st.subheader("From URLs")

    st.code(
        """
import pandas as pd

# Direct CSV from URL
url = "https://example.com/data.csv"
df = pd.read_csv(url)

# OpenML datasets
from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", version=1, as_frame=True)
X, y = mnist.data, mnist.target
    """,
        language="python",
    )

    st.subheader("From SQL")

    st.code(
        """
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:pass@host/db")
df = pd.read_sql("SELECT * FROM table", engine)
    """,
        language="python",
    )


if __name__ == "__main__":
    main()
