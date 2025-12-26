"""Dataset Transformations.

This page covers:
- Pipelines and ColumnTransformer
- Feature extraction
- Preprocessing
- Imputation
"""

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dataset Transformations", page_icon="🔄", layout="wide")

from skplay.core.datasets import get_dataset
from skplay.ui.level import get_level, level_selector


def main():
    st.title("🔄 Dataset Transformations")

    with st.sidebar:
        level_selector()

    st.markdown("""
    Transform raw data into features suitable for machine learning.

    [📚 sklearn User Guide: Dataset Transformations](https://scikit-learn.org/stable/data_transforms.html)
    """)

    topic = st.radio(
        "Topic",
        options=["pipelines", "preprocessing", "imputation", "encoding", "feature_extraction"],
        format_func=lambda x: x.replace("_", " ").title(),
        horizontal=True,
    )

    st.markdown("---")

    if topic == "pipelines":
        pipelines_section()
    elif topic == "preprocessing":
        preprocessing_section()
    elif topic == "imputation":
        imputation_section()
    elif topic == "encoding":
        encoding_section()
    else:
        feature_extraction_section()


def pipelines_section():
    """Pipelines and ColumnTransformer."""
    st.header("Pipelines and ColumnTransformer")

    st.markdown("""
    Pipelines chain multiple transformers and estimators. ColumnTransformer applies
    different transformations to different column subsets.

    **Why use pipelines?**
    - Avoid data leakage (preprocessing fit on training data only)
    - Cleaner code (single .fit()/.predict())
    - Easier deployment (single object to serialize)
    - Works with cross-validation
    """)

    st.subheader("Basic Pipeline")

    st.code(
        """
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Chain transformer -> estimator
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression()),
])

# Fit and predict like any estimator
pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)

# Access individual steps
pipe.named_steps["scaler"]
pipe.named_steps["classifier"]
    """,
        language="python",
    )

    st.subheader("ColumnTransformer")

    st.markdown("""
    Apply different transformations to different column types:
    """)

    st.code(
        """
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define column groups
numeric_features = ["age", "income", "score"]
categorical_features = ["gender", "city", "category"]

# Create preprocessing pipelines for each type
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

# Combine with ColumnTransformer
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# Full pipeline with estimator
full_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression()),
])
    """,
        language="python",
    )

    # Interactive demo
    st.subheader("Interactive Demo")

    if st.button("Run Pipeline Demo", key="pipe_demo"):
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        data = get_dataset("retail_demand")
        X, y = data.X, data.y

        # For demo, make it classification
        y_class = pd.Series(["high" if v > y.median() else "low" for v in y])

        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

        st.write("**Numeric features:**", numeric_features)
        st.write("**Categorical features:**", categorical_features)

        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        full_pipe = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(n_estimators=50, random_state=42)),
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_class, test_size=0.2, random_state=42
        )

        full_pipe.fit(X_train, y_train)
        score = full_pipe.score(X_test, y_test)

        st.success(f"Pipeline trained! Test accuracy: {score:.4f}")

        # Show transformed shape
        X_transformed = preprocessor.fit_transform(X_train)
        st.write(f"**Original shape:** {X_train.shape}")
        st.write(f"**Transformed shape:** {X_transformed.shape}")


def preprocessing_section():
    """Feature preprocessing."""
    st.header("Preprocessing: Scaling and Normalization")

    st.markdown("""
    Many algorithms work better when features are on similar scales.
    """)

    scalers_data = [
        {
            "Scaler": "StandardScaler",
            "Formula": "(x - mean) / std",
            "Range": "~(-3, 3)",
            "Use When": "Normal distribution, most algorithms",
        },
        {
            "Scaler": "MinMaxScaler",
            "Formula": "(x - min) / (max - min)",
            "Range": "[0, 1]",
            "Use When": "Need bounded values, neural networks",
        },
        {
            "Scaler": "RobustScaler",
            "Formula": "(x - median) / IQR",
            "Range": "Varies",
            "Use When": "Data has outliers",
        },
        {
            "Scaler": "MaxAbsScaler",
            "Formula": "x / max(|x|)",
            "Range": "[-1, 1]",
            "Use When": "Sparse data, preserve zeros",
        },
        {
            "Scaler": "Normalizer",
            "Formula": "x / ||x||",
            "Range": "Unit norm",
            "Use When": "Text, TF-IDF vectors",
        },
    ]

    st.dataframe(pd.DataFrame(scalers_data), hide_index=True, use_container_width=True)

    # Interactive demo
    st.subheader("Interactive Demo")

    col1, col2 = st.columns([1, 2])

    with col1:
        scaler_choice = st.selectbox(
            "Scaler", ["StandardScaler", "MinMaxScaler", "RobustScaler"], key="scaler_choice"
        )

    with col2:
        if st.button("Apply Scaler", key="scaler_demo"):
            import matplotlib.pyplot as plt
            from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

            data = get_dataset("california_housing")
            X = data.X

            # Select a feature
            feature = X.columns[0]
            original = X[feature].values.reshape(-1, 1)

            scalers = {
                "StandardScaler": StandardScaler(),
                "MinMaxScaler": MinMaxScaler(),
                "RobustScaler": RobustScaler(),
            }

            scaler = scalers[scaler_choice]
            transformed = scaler.fit_transform(original)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].hist(original, bins=50, edgecolor="black")
            axes[0].set_title(f"Original: {feature}")
            axes[0].set_xlabel("Value")

            axes[1].hist(transformed, bins=50, edgecolor="black", color="orange")
            axes[1].set_title(f"After {scaler_choice}")
            axes[1].set_xlabel("Value")

            st.pyplot(fig)

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Original:** mean={original.mean():.2f}, std={original.std():.2f}")
            with col2:
                st.write(
                    f"**Transformed:** mean={transformed.mean():.2f}, std={transformed.std():.2f}"
                )


def imputation_section():
    """Missing value imputation."""
    st.header("Imputation: Handling Missing Values")

    st.markdown("""
    Real-world data often has missing values. Imputation fills these gaps.
    """)

    imputers_data = [
        {
            "Method": "mean",
            "Description": "Replace with column mean",
            "When": "Numeric, normal distribution",
        },
        {
            "Method": "median",
            "Description": "Replace with column median",
            "When": "Numeric, outliers present",
        },
        {
            "Method": "most_frequent",
            "Description": "Replace with mode",
            "When": "Categorical or discrete numeric",
        },
        {
            "Method": "constant",
            "Description": "Replace with fixed value",
            "When": "Need explicit missing indicator",
        },
        {
            "Method": "KNNImputer",
            "Description": "Use k-nearest neighbors",
            "When": "Correlated features",
        },
        {
            "Method": "IterativeImputer",
            "Description": "Multivariate imputation",
            "When": "Complex missing patterns",
        },
    ]

    st.dataframe(pd.DataFrame(imputers_data), hide_index=True, use_container_width=True)

    st.code(
        """
from sklearn.impute import SimpleImputer, KNNImputer

# Simple strategies
imputer = SimpleImputer(strategy="median")  # or "mean", "most_frequent", "constant"
X_imputed = imputer.fit_transform(X)

# K-nearest neighbors
knn_imputer = KNNImputer(n_neighbors=5)
X_imputed = knn_imputer.fit_transform(X)

# Iterative (experimental)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

iter_imputer = IterativeImputer(max_iter=10, random_state=42)
X_imputed = iter_imputer.fit_transform(X)
    """,
        language="python",
    )


def encoding_section():
    """Categorical encoding."""
    st.header("Encoding: Categorical Variables")

    st.markdown("""
    Machine learning algorithms need numeric inputs. Encode categorical variables appropriately.
    """)

    encoders_data = [
        {
            "Encoder": "OneHotEncoder",
            "Output": "Binary columns",
            "When": "Nominal categories, tree-based ok, linear models",
        },
        {
            "Encoder": "OrdinalEncoder",
            "Output": "Integer codes",
            "When": "Ordinal categories with natural order",
        },
        {"Encoder": "LabelEncoder", "Output": "Integer codes", "When": "Target variable only"},
        {
            "Encoder": "TargetEncoder",
            "Output": "Target mean",
            "When": "High cardinality, prevent overfitting",
        },
    ]

    st.dataframe(pd.DataFrame(encoders_data), hide_index=True, use_container_width=True)

    st.code(
        """
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

# One-hot encoding
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_encoded = ohe.fit_transform(X[["category"]])

# Get feature names
ohe.get_feature_names_out(["category"])

# Ordinal encoding (for ordered categories)
oe = OrdinalEncoder(categories=[["low", "medium", "high"]])
X_encoded = oe.fit_transform(X[["priority"]])

# Label encoding (for target)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
    """,
        language="python",
    )

    st.subheader("Interactive Demo")

    if st.button("Encoding Demo", key="enc_demo"):
        from sklearn.preprocessing import OneHotEncoder

        # Create sample categorical data
        sample = pd.DataFrame(
            {
                "color": ["red", "blue", "green", "red", "blue"],
                "size": ["S", "M", "L", "M", "S"],
            }
        )

        st.write("**Original Data:**")
        st.dataframe(sample)

        ohe = OneHotEncoder(sparse_output=False)
        encoded = ohe.fit_transform(sample)
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(["color", "size"]))

        st.write("**After One-Hot Encoding:**")
        st.dataframe(encoded_df)


def feature_extraction_section():
    """Feature extraction."""
    st.header("Feature Extraction")

    level = get_level()

    st.markdown("""
    Extract features from raw data like text, images, or dictionaries.
    """)

    st.subheader("Text Features")

    st.code(
        """
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Bag of words
count_vec = CountVectorizer(max_features=1000, stop_words="english")
X_counts = count_vec.fit_transform(documents)

# TF-IDF (term frequency-inverse document frequency)
tfidf_vec = TfidfVectorizer(max_features=1000, stop_words="english")
X_tfidf = tfidf_vec.fit_transform(documents)
    """,
        language="python",
    )

    st.subheader("Dictionary Features")

    st.code(
        """
from sklearn.feature_extraction import DictVectorizer

# Convert list of dicts to feature matrix
vec = DictVectorizer(sparse=False)
X = vec.fit_transform([
    {"city": "NY", "temp": 20},
    {"city": "LA", "temp": 25},
])
    """,
        language="python",
    )

    if level in ("intermediate", "advanced"):
        st.subheader("Polynomial Features")

        st.code(
            """
from sklearn.preprocessing import PolynomialFeatures

# Create polynomial and interaction features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# For [a, b], creates [a, b, a², ab, b²]
        """,
            language="python",
        )


if __name__ == "__main__":
    main()
