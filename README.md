# scikit-learn Playground 🧪

An interactive Streamlit web application for learning and experimenting with scikit-learn. The app follows the structure of the official [sklearn User Guide](https://scikit-learn.org/stable/user_guide.html) and provides hands-on experience with machine learning concepts.

## Features

- 📚 **User Guide Coverage**: Interactive pages for all major sklearn User Guide sections
- 🎚️ **Three Experience Levels**: Beginner, Intermediate, and Advanced modes
- 📊 **Domain-Specific Datasets**: Power, Retail, Finance, Healthcare, and General
- 📤 **CSV Upload**: Use your own data
- 🔧 **Pipeline Builder**: Visual preprocessing and model configuration
- 📈 **Rich Visualizations**: Learning curves, confusion matrices, feature importance
- 💾 **Export**: Download trained models and reproducible code snippets
- 🔎 **API Explorer**: Search and explore sklearn classes and functions

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/example/sklearn-playground.git
cd sklearn-playground

# Install dependencies with uv
uv sync

# Run the app
uv run streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Alternative: pip installation

```bash
pip install -e .
streamlit run app.py
```

## Project Structure

```
sklearn-playground/
├── app.py                 # Main Streamlit entry point
├── skplay/
│   ├── core/              # Core playground engine
│   │   ├── datasets.py    # Dataset registry and loaders
│   │   ├── upload.py      # CSV upload handling
│   │   ├── preprocessing.py # ColumnTransformer builders
│   │   ├── estimators.py  # Estimator registry
│   │   ├── tuning.py      # Hyperparameter tuning
│   │   ├── evaluation.py  # Metrics and plots
│   │   ├── snippets.py    # Code generation
│   │   └── api_explorer.py # API introspection
│   └── ui/
│       ├── components.py  # Reusable UI widgets
│       └── level.py       # Level gating logic
├── pages/                 # Streamlit pages (User Guide sections)
│   ├── 01_supervised_learning.py
│   ├── 02_unsupervised_learning.py
│   ├── 03_model_selection.py
│   └── ...
├── tests/                 # pytest tests
├── docs/                  # Documentation
└── pyproject.toml         # Project configuration
```

## User Guide Pages

The app covers all major sklearn User Guide sections:

1. **Supervised Learning** - Classification and regression
2. **Unsupervised Learning** - Clustering and outlier detection
3. **Model Selection** - Cross-validation and hyperparameter tuning
4. **Metadata Routing** - Sample weights and groups
5. **Inspection** - Feature importance and PDPs
6. **Visualizations** - sklearn display utilities
7. **Dataset Transformations** - Pipelines and preprocessing
8. **Dataset Loading** - Built-in and custom datasets
9. **Computing** - Performance and parallelism
10. **Model Persistence** - Saving and loading models
11. **Common Pitfalls** - Best practices
12. **Dispatching** - Array API support
13. **Choosing the Right Estimator** - Algorithm selection guide
14. **External Resources** - Learning materials

## Experience Levels

### 🌱 Beginner
- Guided workflow with essential controls
- Detailed explanations and tooltips
- Safe default parameters
- Focus on understanding concepts

### 🌿 Intermediate
- Cross-validation options
- More hyperparameter controls
- Feature selection and regularization
- Learning curves and validation curves

### 🌳 Advanced
- Full parameter access
- Grid/random/halving search
- Code snippet export
- Model persistence
- Advanced metrics

## Datasets

### Built-in Toy Datasets

| Domain | Classification | Regression | Clustering |
|--------|---------------|------------|------------|
| General | iris, wine, breast_cancer, digits | diabetes, california_housing | digits |
| Power | power_failure | power_efficiency | - |
| Retail | retail_churn | retail_demand | retail_segmentation |
| Finance | credit_risk | - | fraud_detection |
| Healthcare | breast_cancer | diabetes | - |

### Custom Data
Upload any CSV file and select the target column.

## Development

### Running Tests

```bash
uv run pytest
```

### Code Quality

```bash
# Linting
uv run ruff check .

# Formatting
uv run ruff format .

# Type checking (optional)
uv run mypy skplay
```

## Adding New Content

### Adding a New Dataset

1. Open `skplay/core/datasets.py`
2. Create a loader function that returns `DatasetResult`
3. Register with `DatasetRegistry.register()`

```python
def _load_my_dataset() -> DatasetResult:
    # Create your data
    X = pd.DataFrame(...)
    y = pd.Series(...)

    # Define features
    features = [FeatureInfo("col1", "numeric", "Description"), ...]

    # Create card
    card = DatasetCard(
        name="my_dataset",
        description="My dataset description",
        task_type="classification",
        domain="general",
        n_samples=len(X),
        n_features=len(X.columns),
        target_name="target",
        features=features,
    )

    return DatasetResult(X=X, y=y, card=card)

# Register
DatasetRegistry.register("my_dataset", _load_my_dataset, _load_my_dataset().card)
```

### Adding a New Page

1. Create a new file in `pages/` with the format `NN_page_name.py`
2. Follow the existing page structure
3. Use components from `skplay.ui.components`

```python
import streamlit as st
from skplay.ui.level import get_level, level_selector

def main():
    st.title("My New Page")

    with st.sidebar:
        level = level_selector()

    # Your content here

if __name__ == "__main__":
    main()
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [scikit-learn](https://scikit-learn.org/) for the amazing ML library
- [Streamlit](https://streamlit.io/) for the web framework
- The sklearn documentation team for excellent learning materials
