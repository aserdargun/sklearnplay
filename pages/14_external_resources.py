"""External Resources.

Links to external learning materials and documentation.
"""

import streamlit as st

st.set_page_config(page_title="External Resources", page_icon="📚", layout="wide")

from skplay.ui.level import level_selector


def main():
    st.title("📚 External Resources")

    with st.sidebar:
        level_selector()

    st.markdown("""
    Curated resources for learning more about scikit-learn and machine learning.

    [📚 sklearn External Resources](https://scikit-learn.org/stable/related_projects.html)
    """)

    st.markdown("---")

    # Official Resources
    st.header("Official scikit-learn Resources")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Documentation
        - [User Guide](https://scikit-learn.org/stable/user_guide.html) - Comprehensive tutorials
        - [API Reference](https://scikit-learn.org/stable/api/index.html) - Complete API docs
        - [Examples Gallery](https://scikit-learn.org/stable/auto_examples/index.html) - Code examples
        - [FAQ](https://scikit-learn.org/stable/faq.html) - Frequently asked questions
        """)

    with col2:
        st.markdown("""
        ### Community
        - [GitHub](https://github.com/scikit-learn/scikit-learn) - Source code and issues
        - [Mailing List](https://mail.python.org/mailman/listinfo/scikit-learn) - Discussions
        - [Stack Overflow](https://stackoverflow.com/questions/tagged/scikit-learn) - Q&A
        - [Gitter Chat](https://gitter.im/scikit-learn/scikit-learn) - Real-time chat
        """)

    st.markdown("---")

    # Related Projects
    st.header("Related Projects")

    st.markdown("""
    ### Ecosystem Libraries

    | Library | Purpose | Link |
    |---------|---------|------|
    | **imbalanced-learn** | Handling imbalanced datasets | [imbalanced-learn.org](https://imbalanced-learn.org) |
    | **category_encoders** | Categorical encoding | [GitHub](https://github.com/scikit-learn-contrib/category_encoders) |
    | **sklearn-pandas** | Integration with pandas | [GitHub](https://github.com/scikit-learn-contrib/sklearn-pandas) |
    | **mlxtend** | Extensions for sklearn | [mlxtend.github.io](https://mlxtend.github.io) |
    | **skops** | Model sharing and deployment | [GitHub](https://github.com/skops-dev/skops) |
    | **SHAP** | Model explanations | [shap.readthedocs.io](https://shap.readthedocs.io) |
    """)

    st.markdown("""
    ### AutoML and Hyperparameter Tuning

    | Library | Purpose | Link |
    |---------|---------|------|
    | **auto-sklearn** | Automated machine learning | [automl.github.io/auto-sklearn](https://automl.github.io/auto-sklearn) |
    | **Optuna** | Hyperparameter optimization | [optuna.org](https://optuna.org) |
    | **Hyperopt** | Distributed hyperparameter search | [hyperopt.github.io](https://hyperopt.github.io) |
    | **TPOT** | Automated pipeline optimization | [epistasislab.github.io/tpot](https://epistasislab.github.io/tpot) |
    """)

    st.markdown("""
    ### Deep Learning Integration

    | Library | Purpose | Link |
    |---------|---------|------|
    | **scikeras** | Keras wrappers for sklearn | [GitHub](https://github.com/adriangb/scikeras) |
    | **skorch** | PyTorch wrappers for sklearn | [skorch.readthedocs.io](https://skorch.readthedocs.io) |
    """)

    st.markdown("---")

    # Learning Resources
    st.header("Learning Resources")

    st.subheader("Books")

    st.markdown("""
    - **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow** by Aurélien Géron
    - **Python Machine Learning** by Sebastian Raschka
    - **Introduction to Machine Learning with Python** by Andreas C. Müller & Sarah Guido
    - **The Elements of Statistical Learning** by Hastie, Tibshirani, and Friedman (free online)
    """)

    st.subheader("Online Courses")

    st.markdown("""
    - [Coursera: Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning) - Classic introduction
    - [Fast.ai](https://www.fast.ai/) - Practical deep learning
    - [Kaggle Learn](https://www.kaggle.com/learn) - Free micro-courses
    - [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course) - Quick overview
    """)

    st.subheader("Video Tutorials")

    st.markdown("""
    - [scikit-learn YouTube](https://www.youtube.com/c/sciikiiearn) - Official channel
    - [StatQuest](https://www.youtube.com/c/joshstarmer) - Clear ML explanations
    - [3Blue1Brown](https://www.youtube.com/c/3blue1brown) - Beautiful math visualizations
    """)

    st.markdown("---")

    # Practice
    st.header("Practice and Competitions")

    st.markdown("""
    - [Kaggle](https://www.kaggle.com/) - Competitions and datasets
    - [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) - Classic datasets
    - [OpenML](https://www.openml.org/) - Collaborative ML platform
    - [DrivenData](https://www.drivendata.org/) - Data science competitions for social good
    """)


if __name__ == "__main__":
    main()
