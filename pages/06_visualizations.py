"""Visualizations.

This page covers sklearn's built-in visualization tools and displays.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Visualizations", page_icon="📊", layout="wide")

from skplay.core.datasets import get_dataset
from skplay.ui.level import get_level, level_selector


def main():
    st.title("📊 Visualizations")

    with st.sidebar:
        level = level_selector()

    st.markdown("""
    sklearn provides built-in visualization tools through the `Display` classes.
    These provide consistent, publication-ready plots for common ML visualizations.

    [📚 sklearn User Guide: Visualizations](https://scikit-learn.org/stable/visualizations.html)
    """)

    viz_type = st.radio(
        "Visualization Type",
        options=["confusion_matrix", "roc_curve", "precision_recall", "learning_curve", "decision_boundary"],
        format_func=lambda x: x.replace("_", " ").title(),
        horizontal=True,
    )

    st.markdown("---")

    if viz_type == "confusion_matrix":
        confusion_matrix_demo()
    elif viz_type == "roc_curve":
        roc_curve_demo()
    elif viz_type == "precision_recall":
        precision_recall_demo()
    elif viz_type == "learning_curve":
        learning_curve_demo()
    else:
        decision_boundary_demo()


def confusion_matrix_demo():
    """Confusion matrix visualization."""
    st.header("Confusion Matrix Display")

    st.markdown("""
    Visualize classification performance with a heatmap of true vs predicted labels.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        dataset = st.selectbox("Dataset", ["iris", "breast_cancer", "wine"], key="cm_data")
        normalize = st.selectbox(
            "Normalize",
            options=[None, "true", "pred", "all"],
            format_func=lambda x: "No" if x is None else x.title(),
            key="cm_norm"
        )

    with col2:
        if st.button("Generate", key="cm_run"):
            with st.spinner("Training..."):
                from sklearn.metrics import ConfusionMatrixDisplay
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.preprocessing import LabelEncoder

                data = get_dataset(dataset)
                X, y = data.X, data.y

                le = LabelEncoder()
                y_encoded = le.fit_transform(y)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=0.3, random_state=42
                )

                model = RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)

                fig, ax = plt.subplots(figsize=(8, 6))
                ConfusionMatrixDisplay.from_estimator(
                    model, X_test, y_test,
                    display_labels=le.classes_,
                    normalize=normalize,
                    ax=ax,
                    cmap="Blues",
                )
                st.pyplot(fig)

    st.code("""
from sklearn.metrics import ConfusionMatrixDisplay

# From estimator (model, X_test, y_test)
ConfusionMatrixDisplay.from_estimator(
    model, X_test, y_test,
    display_labels=class_names,
    normalize='true',  # or 'pred', 'all', None
)

# From predictions (y_true, y_pred)
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=class_names,
)
    """, language="python")


def roc_curve_demo():
    """ROC curve visualization."""
    st.header("ROC Curve Display")

    st.markdown("""
    Receiver Operating Characteristic curve shows trade-off between
    true positive rate and false positive rate at various thresholds.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        dataset = st.selectbox("Dataset", ["breast_cancer"], key="roc_data")
        compare_models = st.checkbox("Compare Multiple Models", key="roc_compare")

    with col2:
        if st.button("Generate", key="roc_run"):
            with st.spinner("Training..."):
                from sklearn.metrics import RocCurveDisplay
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.svm import SVC

                data = get_dataset(dataset)
                X, y = data.X, data.y

                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=0.3, random_state=42
                )

                fig, ax = plt.subplots(figsize=(8, 6))

                if compare_models:
                    models = [
                        ("Random Forest", RandomForestClassifier(random_state=42)),
                        ("Logistic Regression", LogisticRegression(max_iter=1000)),
                        ("SVM", SVC(probability=True, random_state=42)),
                    ]

                    for name, model in models:
                        model.fit(X_train, y_train)
                        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
                else:
                    model = RandomForestClassifier(random_state=42)
                    model.fit(X_train, y_train)
                    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)

                ax.plot([0, 1], [0, 1], 'k--', label='Random')
                ax.legend()
                st.pyplot(fig)

    st.code("""
from sklearn.metrics import RocCurveDisplay

# From estimator
RocCurveDisplay.from_estimator(model, X_test, y_test)

# From predictions
RocCurveDisplay.from_predictions(y_test, y_score)

# Compare multiple models
fig, ax = plt.subplots()
for name, model in models:
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    """, language="python")


def precision_recall_demo():
    """Precision-Recall curve visualization."""
    st.header("Precision-Recall Curve Display")

    st.markdown("""
    More informative than ROC for imbalanced datasets.
    Shows trade-off between precision and recall.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        dataset = st.selectbox("Dataset", ["breast_cancer"], key="pr_data")

    with col2:
        if st.button("Generate", key="pr_run"):
            with st.spinner("Training..."):
                from sklearn.metrics import PrecisionRecallDisplay
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.preprocessing import LabelEncoder

                data = get_dataset(dataset)
                X, y = data.X, data.y

                le = LabelEncoder()
                y_encoded = le.fit_transform(y)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=0.3, random_state=42
                )

                model = RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)

                fig, ax = plt.subplots(figsize=(8, 6))
                PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
                st.pyplot(fig)

    st.code("""
from sklearn.metrics import PrecisionRecallDisplay

# From estimator
PrecisionRecallDisplay.from_estimator(model, X_test, y_test)

# From predictions
PrecisionRecallDisplay.from_predictions(y_test, y_score)
    """, language="python")


def learning_curve_demo():
    """Learning curve visualization."""
    st.header("Learning Curve Display")

    st.markdown("""
    Shows how training and validation scores change with training set size.
    Useful for diagnosing bias/variance issues.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        dataset = st.selectbox("Dataset", ["iris", "breast_cancer"], key="lc_data")

    with col2:
        if st.button("Generate", key="lc_run"):
            with st.spinner("Computing (this may take a moment)..."):
                from sklearn.model_selection import LearningCurveDisplay
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.preprocessing import LabelEncoder

                data = get_dataset(dataset)
                X, y = data.X, data.y

                le = LabelEncoder()
                y_encoded = le.fit_transform(y)

                model = RandomForestClassifier(random_state=42)

                fig, ax = plt.subplots(figsize=(10, 6))
                LearningCurveDisplay.from_estimator(
                    model, X, y_encoded,
                    cv=5,
                    train_sizes=np.linspace(0.1, 1.0, 10),
                    ax=ax,
                    n_jobs=-1,
                )
                st.pyplot(fig)

    st.code("""
from sklearn.model_selection import LearningCurveDisplay

LearningCurveDisplay.from_estimator(
    model, X, y,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
)
    """, language="python")


def decision_boundary_demo():
    """Decision boundary visualization."""
    st.header("Decision Boundary Display")

    st.markdown("""
    Visualize how a classifier partitions the feature space (for 2D data).
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        classifier = st.selectbox(
            "Classifier",
            ["Random Forest", "Logistic Regression", "SVM", "KNN"],
            key="db_clf"
        )

    with col2:
        if st.button("Generate", key="db_run"):
            with st.spinner("Training..."):
                from sklearn.inspection import DecisionBoundaryDisplay
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.svm import SVC
                from sklearn.neighbors import KNeighborsClassifier

                # Use iris with first 2 features
                data = get_dataset("iris")
                X = data.X.iloc[:, :2].values
                y = data.y

                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)

                models = {
                    "Random Forest": RandomForestClassifier(random_state=42),
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "SVM": SVC(kernel='rbf', random_state=42),
                    "KNN": KNeighborsClassifier(n_neighbors=5),
                }

                model = models[classifier]
                model.fit(X, y_encoded)

                fig, ax = plt.subplots(figsize=(10, 8))
                DecisionBoundaryDisplay.from_estimator(
                    model, X, ax=ax,
                    response_method="predict",
                    alpha=0.5,
                    cmap="viridis",
                )
                scatter = ax.scatter(X[:, 0], X[:, 1], c=y_encoded, edgecolor='black', cmap="viridis")
                ax.set_xlabel(data.X.columns[0])
                ax.set_ylabel(data.X.columns[1])
                ax.set_title(f"{classifier} Decision Boundary (Iris: first 2 features)")
                st.pyplot(fig)

    st.code("""
from sklearn.inspection import DecisionBoundaryDisplay

DecisionBoundaryDisplay.from_estimator(
    model, X,
    response_method="predict",  # or "predict_proba"
    alpha=0.5,
)
    """, language="python")


if __name__ == "__main__":
    main()
