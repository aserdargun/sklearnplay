"""Dispatching.

This page covers array API dispatching for different backends.
"""

import streamlit as st

st.set_page_config(page_title="Dispatching", page_icon="🔌", layout="wide")

from skplay.ui.level import get_level, level_selector


def main():
    st.title("🔌 Dispatching")

    with st.sidebar:
        level = level_selector()

    st.markdown("""
    Array API dispatching allows sklearn to work with different array backends
    (NumPy, CuPy, PyTorch, etc.) for hardware acceleration.

    [📚 sklearn User Guide: Array API](https://scikit-learn.org/stable/modules/array_api.html)
    """)

    if level == "beginner":
        st.info("""
        This is an advanced topic for users who want to leverage GPUs or
        specialized hardware. Most users can skip this section.
        """)

        st.markdown("""
        ### Quick Summary

        By default, sklearn uses NumPy arrays (CPU). With Array API support,
        you can use GPU arrays (CuPy, PyTorch) for faster computation on
        certain algorithms.

        **When you might need this:**
        - Processing very large datasets
        - Need GPU acceleration
        - Integrating sklearn into a PyTorch pipeline
        """)

    else:
        st.markdown("""
        ### Array API Overview

        The Python Array API standard provides a common interface for array
        operations across different libraries. sklearn is gradually adding
        support for this standard.
        """)

        st.subheader("Enabling Array API")

        st.code("""
import sklearn
from sklearn import config_context

# Enable Array API dispatch globally
sklearn.set_config(array_api_dispatch=True)

# Or for a specific context
with config_context(array_api_dispatch=True):
    model.fit(X, y)  # Uses the array's native backend
        """, language="python")

        st.subheader("Using with CuPy (GPU)")

        st.code("""
import cupy as cp
import sklearn
from sklearn.linear_model import Ridge

sklearn.set_config(array_api_dispatch=True)

# Convert data to CuPy arrays (GPU)
X_gpu = cp.asarray(X)
y_gpu = cp.asarray(y)

# Fit on GPU
model = Ridge()
model.fit(X_gpu, y_gpu)

# Predictions stay on GPU
predictions = model.predict(X_gpu)

# Convert back to NumPy if needed
predictions_cpu = cp.asnumpy(predictions)
        """, language="python")

        st.subheader("Using with PyTorch")

        st.code("""
import torch
import sklearn
from sklearn.decomposition import PCA

sklearn.set_config(array_api_dispatch=True)

# Convert to PyTorch tensors
X_torch = torch.tensor(X, device="cuda")

# PCA on GPU
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X_torch)

# Result is a PyTorch tensor on GPU
print(type(X_reduced))  # torch.Tensor
        """, language="python")

        st.subheader("Supported Estimators")

        st.markdown("""
        Not all sklearn estimators support Array API yet. Check the documentation
        for each estimator. Generally supported:

        - **Preprocessing**: StandardScaler, MinMaxScaler
        - **Decomposition**: PCA
        - **Linear Models**: Ridge, LogisticRegression (some solvers)
        - **Neighbors**: KNeighborsClassifier (some metrics)

        Support is being actively expanded in newer versions.
        """)

        st.subheader("Considerations")

        st.warning("""
        **Things to keep in mind:**

        - Data transfer between CPU and GPU has overhead
        - Not all estimators support all backends
        - GPU memory is limited
        - Some operations may fall back to CPU
        - Debug on CPU first, then move to GPU
        """)

        if level == "advanced":
            st.subheader("Checking Backend Support")

            st.code("""
import sklearn.utils._array_api as array_api

# Check if estimator supports array API
from sklearn.linear_model import Ridge
ridge = Ridge()

# This is an internal check (may change)
# Best practice: check documentation or try/except
try:
    with sklearn.config_context(array_api_dispatch=True):
        ridge.fit(X_gpu, y_gpu)
except Exception as e:
    print(f"Array API not supported: {e}")
            """, language="python")


if __name__ == "__main__":
    main()
