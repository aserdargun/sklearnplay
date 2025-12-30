"""Metadata Routing.

This page covers:
- Sample weights routing
- Groups routing
- Custom metadata passing
"""

import streamlit as st

st.set_page_config(page_title="Metadata Routing", page_icon="images/icon.png", layout="wide")

from skplay.ui.level import level_selector


def main():
    st.title("🔀 Metadata Routing")

    with st.sidebar:
        level = level_selector()

    st.markdown("""
    Metadata routing (introduced in sklearn 1.3+) provides a mechanism to pass
    additional information (like sample weights or groups) through pipelines and meta-estimators.

    [📚 sklearn User Guide: Metadata Routing](https://scikit-learn.org/stable/metadata_routing.html)
    """)

    if level == "beginner":
        st.info(
            "This is an advanced topic. Switch to Intermediate or Advanced level for full content."
        )

        st.markdown("""
        ### Quick Summary

        Metadata routing solves the problem of passing extra information (like sample weights)
        through complex pipelines. Before this feature, it was difficult to use sample weights
        with cross-validation or nested pipelines.

        **When you need it:**
        - Using sample weights with cross-validation
        - Passing group information for GroupKFold
        - Custom scoring that needs additional data
        """)

    else:
        st.markdown("""
        ### Understanding Metadata Routing

        In machine learning workflows, we often need to pass additional information beyond
        X (features) and y (targets). Common examples include:

        - **Sample weights**: Importance of each sample
        - **Groups**: Which samples belong together (e.g., same patient)
        - **Fit params**: Parameters for specific estimator methods

        The challenge is routing this metadata through complex compositions like pipelines,
        cross-validation, or meta-estimators (e.g., BaggingClassifier).
        """)

        st.markdown("---")
        st.subheader("Basic Concept")

        st.markdown("""
        Metadata routing works by:

        1. **Configuring routing** on estimators/scorers to specify which metadata they consume
        2. **Enabling routing** globally
        3. **Passing metadata** to the fit/score methods
        """)

        # Code example
        st.code(
            """
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

# Enable metadata routing globally
sklearn.set_config(enable_metadata_routing=True)

# Configure estimator to accept sample_weight
lr = LogisticRegression()
lr.set_fit_request(sample_weight=True)

# Now sample_weight will be passed through cross_validate
results = cross_validate(
    lr, X, y,
    cv=5,
    params={"sample_weight": weights}
)
        """,
            language="python",
        )

        st.markdown("---")
        st.subheader("Routing Through Pipelines")

        st.code(
            """
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Create pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression()),
])

# Configure the classifier step to accept sample_weight
pipe.set_fit_request(clf__sample_weight=True)

# Fit with sample weights
pipe.fit(X, y, clf__sample_weight=weights)
        """,
            language="python",
        )

        st.markdown("---")
        st.subheader("Routing with Cross-Validation")

        st.code(
            """
from sklearn.model_selection import cross_validate, GroupKFold
from sklearn.linear_model import Ridge

# Configure estimator
ridge = Ridge()
ridge.set_fit_request(sample_weight=True)

# Use GroupKFold which needs groups
cv = GroupKFold(n_splits=5)

# Pass both sample_weight and groups
results = cross_validate(
    ridge, X, y,
    cv=cv,
    params={
        "sample_weight": weights,
        "groups": groups,  # For GroupKFold
    }
)
        """,
            language="python",
        )

        if level == "advanced":
            st.markdown("---")
            st.subheader("Custom Routing Configuration")

            st.code(
                """
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _routing_enabled,
)

class CustomEstimator(BaseEstimator, ClassifierMixin):
    def get_metadata_routing(self):
        # Define what metadata this estimator uses
        router = MetadataRouter(owner=self.__class__.__name__)

        # Add routing for sample_weight in fit
        router.add_self_request(
            self,
            method=MethodMapping()
                .add(callee="fit", caller="fit")
        )

        return router

    def fit(self, X, y, sample_weight=None):
        # Use sample_weight if provided
        if sample_weight is not None:
            # Apply weights to fitting
            pass
        return self
            """,
                language="python",
            )

        # Best practices
        st.markdown("---")
        st.subheader("Best Practices")

        st.markdown("""
        1. **Enable routing explicitly** at the start of your script
        2. **Configure all estimators** that need metadata before using them
        3. **Use consistent naming** for metadata across your pipeline
        4. **Document metadata requirements** in your custom estimators
        5. **Test routing** with simple examples before complex pipelines
        """)

        # Interactive demo placeholder
        st.markdown("---")
        st.subheader("Interactive Demo")

        st.info("""
        Metadata routing requires sklearn 1.3+ and explicit configuration.
        Here's a conceptual demonstration of how sample weights affect model training.
        """)

        import numpy as np

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Without Sample Weights:**")
            st.markdown("All samples treated equally")

        with col2:
            st.markdown("**With Sample Weights:**")
            st.markdown("Important samples influence model more")

        # Simple visualization
        import matplotlib.pyplot as plt

        np.random.seed(42)
        X_demo = np.random.randn(50, 1)
        y_demo = 2 * X_demo.ravel() + np.random.randn(50) * 0.5

        # Add some outliers
        X_demo = np.vstack([X_demo, [[3], [3.5]]])
        y_demo = np.append(y_demo, [-2, -3])  # Outliers

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Without weights
        from sklearn.linear_model import LinearRegression

        lr1 = LinearRegression()
        lr1.fit(X_demo, y_demo)

        axes[0].scatter(X_demo, y_demo)
        axes[0].plot(X_demo, lr1.predict(X_demo), "r-", linewidth=2)
        axes[0].set_title("Without Sample Weights")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("y")

        # With weights (downweight outliers)
        weights = np.ones(len(y_demo))
        weights[-2:] = 0.1  # Downweight outliers

        lr2 = LinearRegression()
        lr2.fit(X_demo, y_demo, sample_weight=weights)

        axes[1].scatter(X_demo, y_demo, c=weights, cmap="RdYlGn", vmin=0, vmax=1)
        axes[1].plot(X_demo, lr2.predict(X_demo), "r-", linewidth=2)
        axes[1].set_title("With Sample Weights (outliers downweighted)")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("y")

        st.pyplot(fig)

        st.caption(
            "Green = high weight, Red = low weight. Notice how the line fits the majority better with weights."
        )


if __name__ == "__main__":
    main()
