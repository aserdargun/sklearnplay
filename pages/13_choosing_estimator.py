"""Choosing the Right Estimator.

This page helps users select appropriate algorithms for their problem.
"""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Choosing Estimator", page_icon="images/icon.png", layout="wide")

from skplay.ui.level import level_selector


def main():
    st.title("🧭 Choosing the Right Estimator")

    with st.sidebar:
        level_selector()

    st.markdown("""
    Select the right algorithm based on your problem type, data characteristics,
    and requirements.

    [📚 sklearn: Choosing the Right Estimator](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
    """)

    # Decision flow
    st.subheader("Quick Decision Flow")

    st.markdown("""
    ### Step 1: What type of problem?

    - **Predicting a category?** → Classification
    - **Predicting a number?** → Regression
    - **Finding groups?** → Clustering
    - **Finding unusual points?** → Outlier Detection
    - **Reducing dimensions?** → Dimensionality Reduction
    """)

    problem_type = st.radio(
        "Select your problem type",
        options=[
            "classification",
            "regression",
            "clustering",
            "outlier_detection",
            "dimensionality_reduction",
        ],
        format_func=lambda x: x.replace("_", " ").title(),
        horizontal=True,
    )

    st.markdown("---")

    if problem_type == "classification":
        classification_guide()
    elif problem_type == "regression":
        regression_guide()
    elif problem_type == "clustering":
        clustering_guide()
    elif problem_type == "outlier_detection":
        outlier_guide()
    else:
        dim_reduction_guide()


def classification_guide():
    """Guide for classification problems."""
    st.header("Classification Algorithms")

    st.markdown("""
    ### Decision Factors

    Consider:
    1. **Dataset size** (samples × features)
    2. **Need for interpretability?**
    3. **Need for probability estimates?**
    4. **Computational resources**
    5. **Accuracy vs speed tradeoff**
    """)

    # Algorithm recommendations
    algorithms = [
        {
            "Algorithm": "Logistic Regression",
            "Good For": "Baseline, interpretable, linear boundaries",
            "Data Size": "Any",
            "Speed": "Fast",
            "Interpretable": "Yes",
        },
        {
            "Algorithm": "Random Forest",
            "Good For": "General purpose, feature importance",
            "Data Size": "Any",
            "Speed": "Medium",
            "Interpretable": "Moderate",
        },
        {
            "Algorithm": "Gradient Boosting",
            "Good For": "Best accuracy on tabular data",
            "Data Size": "Medium+",
            "Speed": "Slow",
            "Interpretable": "Low",
        },
        {
            "Algorithm": "SVM",
            "Good For": "High-dimensional, complex boundaries",
            "Data Size": "Small-Medium",
            "Speed": "Slow (large n)",
            "Interpretable": "Low",
        },
        {
            "Algorithm": "KNN",
            "Good For": "Simple, no training, lazy learning",
            "Data Size": "Small-Medium",
            "Speed": "Fast train, slow predict",
            "Interpretable": "Yes",
        },
        {
            "Algorithm": "Naive Bayes",
            "Good For": "Text, high dimensions, fast",
            "Data Size": "Any",
            "Speed": "Very Fast",
            "Interpretable": "Yes",
        },
    ]

    st.dataframe(pd.DataFrame(algorithms), hide_index=True, width="stretch")

    st.markdown("""
    ### Quick Recommendations

    - **Start here**: Logistic Regression → Random Forest
    - **Need best accuracy**: HistGradientBoostingClassifier
    - **Need interpretability**: Decision Tree, Logistic Regression
    - **High-dimensional sparse**: LinearSVC, Naive Bayes
    - **Small dataset**: SVM with RBF kernel
    """)


def regression_guide():
    """Guide for regression problems."""
    st.header("Regression Algorithms")

    algorithms = [
        {
            "Algorithm": "Linear Regression",
            "Good For": "Baseline, interpretable",
            "Data Size": "Any",
            "Speed": "Very Fast",
            "Handles Nonlinearity": "No",
        },
        {
            "Algorithm": "Ridge/Lasso",
            "Good For": "Regularization, feature selection (Lasso)",
            "Data Size": "Any",
            "Speed": "Fast",
            "Handles Nonlinearity": "No",
        },
        {
            "Algorithm": "Random Forest",
            "Good For": "General purpose, nonlinear",
            "Data Size": "Any",
            "Speed": "Medium",
            "Handles Nonlinearity": "Yes",
        },
        {
            "Algorithm": "Gradient Boosting",
            "Good For": "Best accuracy on tabular",
            "Data Size": "Medium+",
            "Speed": "Slow",
            "Handles Nonlinearity": "Yes",
        },
        {
            "Algorithm": "SVR",
            "Good For": "Complex patterns, small data",
            "Data Size": "Small-Medium",
            "Speed": "Slow",
            "Handles Nonlinearity": "Yes (RBF)",
        },
        {
            "Algorithm": "KNeighborsRegressor",
            "Good For": "Local patterns",
            "Data Size": "Small-Medium",
            "Speed": "Fast train, slow predict",
            "Handles Nonlinearity": "Yes",
        },
    ]

    st.dataframe(pd.DataFrame(algorithms), hide_index=True, width="stretch")

    st.markdown("""
    ### Quick Recommendations

    - **Start here**: Ridge → Random Forest
    - **Need best accuracy**: HistGradientBoostingRegressor
    - **Need interpretability**: Linear Regression, Ridge
    - **Feature selection**: Lasso, ElasticNet
    - **Robust to outliers**: HuberRegressor
    """)


def clustering_guide():
    """Guide for clustering problems."""
    st.header("Clustering Algorithms")

    algorithms = [
        {
            "Algorithm": "K-Means",
            "Good For": "Spherical clusters, scalable",
            "Needs k": "Yes",
            "Handles": "Convex clusters",
            "Speed": "Fast",
        },
        {
            "Algorithm": "DBSCAN",
            "Good For": "Arbitrary shapes, noise detection",
            "Needs k": "No",
            "Handles": "Any shape, noise",
            "Speed": "Medium",
        },
        {
            "Algorithm": "Hierarchical",
            "Good For": "Dendrogram, nested clusters",
            "Needs k": "Optional",
            "Handles": "Any shape",
            "Speed": "Slow (large n)",
        },
        {
            "Algorithm": "Gaussian Mixture",
            "Good For": "Soft clustering, probabilistic",
            "Needs k": "Yes",
            "Handles": "Elliptical",
            "Speed": "Medium",
        },
        {
            "Algorithm": "OPTICS",
            "Good For": "Varying density",
            "Needs k": "No",
            "Handles": "Any shape",
            "Speed": "Slow",
        },
    ]

    st.dataframe(pd.DataFrame(algorithms), hide_index=True, width="stretch")

    st.markdown("""
    ### Quick Recommendations

    - **Start here**: K-Means → DBSCAN
    - **Don't know k**: DBSCAN, OPTICS, Mean Shift
    - **Need probabilities**: Gaussian Mixture
    - **Hierarchical structure**: AgglomerativeClustering
    - **Very large data**: MiniBatchKMeans, BIRCH
    """)


def outlier_guide():
    """Guide for outlier detection."""
    st.header("Outlier Detection Algorithms")

    algorithms = [
        {
            "Algorithm": "Isolation Forest",
            "Good For": "General purpose, high dimensions",
            "Approach": "Isolation-based",
            "Speed": "Fast",
        },
        {
            "Algorithm": "Local Outlier Factor",
            "Good For": "Local outliers, varying density",
            "Approach": "Density-based",
            "Speed": "Medium",
        },
        {
            "Algorithm": "One-Class SVM",
            "Good For": "Clear normal boundary",
            "Approach": "Boundary-based",
            "Speed": "Slow (large n)",
        },
        {
            "Algorithm": "Elliptic Envelope",
            "Good For": "Gaussian-distributed data",
            "Approach": "Covariance-based",
            "Speed": "Fast",
        },
    ]

    st.dataframe(pd.DataFrame(algorithms), hide_index=True, width="stretch")

    st.markdown("""
    ### Quick Recommendations

    - **Start here**: Isolation Forest
    - **Local anomalies**: Local Outlier Factor
    - **Gaussian data**: Elliptic Envelope
    - **Complex boundaries**: One-Class SVM
    """)


def dim_reduction_guide():
    """Guide for dimensionality reduction."""
    st.header("Dimensionality Reduction")

    algorithms = [
        {
            "Algorithm": "PCA",
            "Good For": "General purpose, variance preservation",
            "Type": "Linear",
            "Preserves": "Global structure",
        },
        {
            "Algorithm": "t-SNE",
            "Good For": "Visualization, local structure",
            "Type": "Nonlinear",
            "Preserves": "Local neighborhoods",
        },
        {
            "Algorithm": "UMAP",
            "Good For": "Visualization, faster than t-SNE",
            "Type": "Nonlinear",
            "Preserves": "Local + some global",
        },
        {
            "Algorithm": "Truncated SVD",
            "Good For": "Sparse data, text (LSA)",
            "Type": "Linear",
            "Preserves": "Global structure",
        },
        {
            "Algorithm": "LDA",
            "Good For": "Supervised reduction, classification",
            "Type": "Linear",
            "Preserves": "Class separation",
        },
    ]

    st.dataframe(pd.DataFrame(algorithms), hide_index=True, width="stretch")

    st.markdown("""
    ### Quick Recommendations

    - **Preprocessing**: PCA
    - **Visualization (2D/3D)**: t-SNE, UMAP
    - **Sparse data**: TruncatedSVD
    - **Classification preprocessing**: LDA
    - **Feature extraction**: PCA, Kernel PCA
    """)


if __name__ == "__main__":
    main()
