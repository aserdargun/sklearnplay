"""Common Pitfalls and Recommended Practices.

This page covers mistakes to avoid and best practices.
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Pitfalls & Best Practices", page_icon="⚠️", layout="wide")

from skplay.ui.level import get_level, level_selector


def main():
    st.title("⚠️ Common Pitfalls and Recommended Practices")

    with st.sidebar:
        level = level_selector()

    st.markdown("""
    Learn from common mistakes and follow best practices for reliable ML.

    [📚 sklearn User Guide: Common Pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
    """)

    topic = st.radio(
        "Topic",
        options=["data_leakage", "preprocessing", "evaluation", "general"],
        format_func=lambda x: x.replace("_", " ").title(),
        horizontal=True,
    )

    st.markdown("---")

    if topic == "data_leakage":
        data_leakage_section()
    elif topic == "preprocessing":
        preprocessing_section()
    elif topic == "evaluation":
        evaluation_section()
    else:
        general_section()


def data_leakage_section():
    """Data leakage pitfalls."""
    st.header("Data Leakage")

    st.error("""
    **Data leakage** occurs when information from outside the training set
    influences model training. This leads to overly optimistic evaluation
    that doesn't reflect real-world performance.
    """)

    st.subheader("❌ Wrong: Preprocessing Before Splitting")

    st.code("""
# WRONG - Information leaks from test set to training
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit on ALL data including test!

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
model.fit(X_train, y_train)
    """, language="python")

    st.subheader("✅ Right: Split First, Then Preprocess")

    st.code("""
# CORRECT - Preprocess after splitting
X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit only on training
X_test_scaled = scaler.transform(X_test)  # Transform test (no fitting!)

model.fit(X_train_scaled, y_train)
    """, language="python")

    st.subheader("✅ Best: Use Pipelines")

    st.code("""
# BEST - Pipeline handles it automatically
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Pipeline fits scaler only on training data
pipeline.fit(X_train, y_train)
score = pipeline.score(X_test, y_test)
    """, language="python")

    st.subheader("Other Sources of Leakage")

    leakage_sources = [
        {"Source": "Target leakage", "Example": "Using future data to predict past", "Fix": "Respect temporal order"},
        {"Source": "Feature from target", "Example": "Including target-derived feature", "Fix": "Audit feature engineering"},
        {"Source": "Duplicate data", "Example": "Same sample in train and test", "Fix": "Remove duplicates before split"},
        {"Source": "Group leakage", "Example": "Same patient in train and test", "Fix": "Use GroupKFold"},
    ]

    st.dataframe(pd.DataFrame(leakage_sources), hide_index=True, use_container_width=True)


def preprocessing_section():
    """Preprocessing pitfalls."""
    st.header("Preprocessing Pitfalls")

    st.subheader("❌ Forgetting to Scale")

    st.markdown("""
    Many algorithms are sensitive to feature scales:
    - **SVM, KNN**: Distance-based, affected by scale
    - **Gradient descent**: Converges faster with scaled data
    - **Regularization**: Penalty is scale-dependent
    """)

    st.code("""
# Check if scaling is needed
from sklearn.preprocessing import StandardScaler

# Algorithms that typically need scaling:
# SVC, SVR, KNeighborsClassifier, LogisticRegression (with regularization),
# neural networks, clustering algorithms

# Algorithms that DON'T need scaling:
# Tree-based (RandomForest, GradientBoosting, DecisionTree)
# Naive Bayes
    """, language="python")

    st.subheader("❌ Wrong Encoding for Trees")

    st.markdown("""
    One-hot encoding with many categories can hurt tree-based models.
    Consider ordinal encoding or target encoding for high-cardinality categories.
    """)

    st.subheader("❌ Ignoring Missing Values")

    st.code("""
# Many algorithms can't handle NaN
# Check for missing values
print(X.isnull().sum())

# Options:
# 1. Drop rows with missing values (loses data)
# 2. Impute with SimpleImputer, KNNImputer, etc.
# 3. Use algorithms that handle missing values:
#    - HistGradientBoostingClassifier
#    - HistGradientBoostingRegressor
    """, language="python")


def evaluation_section():
    """Evaluation pitfalls."""
    st.header("Evaluation Pitfalls")

    st.subheader("❌ Using Accuracy with Imbalanced Data")

    st.code("""
# If 95% of samples are class A, predicting all A gives 95% accuracy!

# Better metrics for imbalanced data:
# - Precision, Recall, F1-score
# - ROC AUC, PR AUC
# - Confusion matrix

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
    """, language="python")

    st.subheader("❌ Single Train/Test Split")

    st.markdown("""
    A single split can be misleading. Use cross-validation for more reliable estimates.
    """)

    st.code("""
from sklearn.model_selection import cross_val_score

# Get multiple estimates
scores = cross_val_score(model, X, y, cv=5)
print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    """, language="python")

    st.subheader("❌ Tuning on Test Set")

    st.code("""
# WRONG - Using test set for hyperparameter tuning
for C in [0.1, 1, 10]:
    model = LogisticRegression(C=C)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)  # Tuning on test!
    if score > best_score:
        best_C = C

# Test set is now "contaminated" - final score is optimistic
    """, language="python")

    st.code("""
# CORRECT - Use validation set or cross-validation
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    LogisticRegression(),
    {"C": [0.1, 1, 10]},
    cv=5  # Uses internal validation
)
grid.fit(X_train, y_train)

# Now test set is truly held-out
final_score = grid.score(X_test, y_test)
    """, language="python")


def general_section():
    """General best practices."""
    st.header("General Best Practices")

    practices = [
        ("Always split data first", "Prevent data leakage by separating test set before any preprocessing"),
        ("Use pipelines", "Encapsulate preprocessing and modeling for cleaner, safer code"),
        ("Set random_state", "For reproducibility, always set random seeds"),
        ("Start simple", "Begin with linear models, add complexity only if needed"),
        ("Check assumptions", "Verify data quality, distribution, and feature relevance"),
        ("Cross-validate", "Don't trust single train/test splits"),
        ("Document everything", "Record hyperparameters, preprocessing steps, versions"),
        ("Version control", "Track code, data, and model changes"),
        ("Monitor production", "Check for data drift and model degradation"),
    ]

    for practice, description in practices:
        st.markdown(f"### ✅ {practice}")
        st.markdown(description)

    st.subheader("Reproducibility Checklist")

    st.code("""
import numpy as np
import sklearn

# Set random seeds
np.random.seed(42)

# Document versions
print(f"sklearn version: {sklearn.__version__}")

# Save configuration
config = {
    "random_state": 42,
    "test_size": 0.2,
    "cv_folds": 5,
    "model_params": model.get_params(),
}

# Save with model
import joblib
joblib.dump({"model": model, "config": config}, "model.joblib")
    """, language="python")


if __name__ == "__main__":
    main()
