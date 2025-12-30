"""Model Persistence.

This page covers saving and loading trained models.
"""

import streamlit as st

st.set_page_config(page_title="Model Persistence", page_icon="images/icon.png", layout="wide")

from skplay.ui.level import get_level, level_selector


def main():
    st.title("💾 Model Persistence")

    with st.sidebar:
        level_selector()

    st.markdown("""
    Save trained models to disk and load them later for predictions.

    [📚 sklearn User Guide: Model Persistence](https://scikit-learn.org/stable/model_persistence.html)
    """)

    topic = st.radio(
        "Topic",
        options=["joblib", "pickle", "skops", "security"],
        format_func=lambda x: x.title() if x != "skops" else "Skops",
        horizontal=True,
    )

    st.markdown("---")

    if topic == "joblib":
        joblib_section()
    elif topic == "pickle":
        pickle_section()
    elif topic == "skops":
        skops_section()
    else:
        security_section()


def joblib_section():
    """Joblib persistence."""
    st.header("Joblib (Recommended)")

    st.markdown("""
    Joblib is the recommended way to persist sklearn models. It's efficient for
    objects containing large numpy arrays.
    """)

    st.code(
        """
import joblib
from sklearn.ensemble import RandomForestClassifier

# Train a model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save to file
joblib.dump(model, "model.joblib")

# Load from file
loaded_model = joblib.load("model.joblib")

# Use for predictions
predictions = loaded_model.predict(X_test)
    """,
        language="python",
    )

    st.subheader("With Compression")

    st.code(
        """
# Save with compression (smaller file, slower save/load)
joblib.dump(model, "model.joblib.gz", compress=3)

# Load compressed file (automatic detection)
model = joblib.load("model.joblib.gz")
    """,
        language="python",
    )

    st.subheader("Saving Full Pipelines")

    st.code(
        """
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Create and train pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression()),
])
pipeline.fit(X_train, y_train)

# Save entire pipeline
joblib.dump(pipeline, "pipeline.joblib")

# Load and use
pipeline = joblib.load("pipeline.joblib")
predictions = pipeline.predict(X_new)
    """,
        language="python",
    )


def pickle_section():
    """Python pickle."""
    st.header("Python Pickle")

    st.markdown("""
    Standard Python serialization. Works but less efficient than joblib for large arrays.
    """)

    st.code(
        """
import pickle

# Save
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Load
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
    """,
        language="python",
    )

    st.warning("""
    **Prefer joblib over pickle for sklearn models.** Joblib is more efficient for
    objects with large numpy arrays (common in ML models).
    """)


def skops_section():
    """Skops for secure sharing."""
    st.header("Skops (Secure Sharing)")

    level = get_level()

    st.markdown("""
    Skops provides a more secure format for sharing models, especially useful for
    publishing to Hugging Face Hub.
    """)

    st.code(
        """
# Install: pip install skops

import skops.io as sio

# Save model
sio.dump(model, "model.skops")

# Load model (shows untrusted types)
unknown_types = sio.get_untrusted_types(file="model.skops")
print(unknown_types)  # Review before loading

# Load with trusted types
model = sio.load("model.skops", trusted=unknown_types)
    """,
        language="python",
    )

    st.subheader("Publishing to Hugging Face")

    if level in ("intermediate", "advanced"):
        st.code(
            """
from skops import hub_utils, card

# Create model card
model_card = card.Card(model)
model_card.metadata.license = "mit"
model_card.add(
    model_description="A Random Forest classifier for iris species"
)

# Push to Hub
hub_utils.push(
    repo_id="username/my-model",
    source="model.skops",
    model_card=model_card,
)
        """,
            language="python",
        )


def security_section():
    """Security considerations."""
    st.header("Security Considerations")

    st.error("""
    ⚠️ **CRITICAL SECURITY WARNING**

    Never load pickled/joblib models from untrusted sources!

    Pickle and joblib files can execute arbitrary code when loaded.
    A malicious model file could:
    - Delete files
    - Install malware
    - Steal credentials
    - Compromise your system
    """)

    st.markdown("""
    ### Safe Practices

    1. **Only load models you created** or from trusted sources
    2. **Verify file integrity** (checksums, signatures)
    3. **Use skops** for untrusted models (allows inspection before loading)
    4. **Sandbox** untrusted model loading in isolated environments
    5. **Document provenance** - track where models come from
    """)

    st.subheader("Version Compatibility")

    st.markdown("""
    Models saved with one version of sklearn may not load correctly with another.

    **Best practices:**
    - Document sklearn version: `sklearn.__version__`
    - Pin dependencies in requirements.txt
    - Test loading after upgrading
    - Consider retraining with new versions
    """)

    st.code(
        """
import sklearn
import joblib

# Save with version info
model_data = {
    "model": model,
    "sklearn_version": sklearn.__version__,
    "feature_names": X.columns.tolist(),
    "target_classes": model.classes_.tolist(),
}
joblib.dump(model_data, "model_with_metadata.joblib")

# Load and check version
data = joblib.load("model_with_metadata.joblib")
if data["sklearn_version"] != sklearn.__version__:
    print(f"Warning: Model trained with {data['sklearn_version']}, "
          f"current version is {sklearn.__version__}")
    """,
        language="python",
    )


if __name__ == "__main__":
    main()
