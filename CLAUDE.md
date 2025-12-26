# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

scikit-learn Playground is an interactive Streamlit web application for learning and experimenting with scikit-learn. It provides hands-on ML experiences at three complexity levels (Beginner, Intermediate, Advanced) covering classification, regression, clustering, and outlier detection.

## Common Commands

```bash
# Install dependencies (using uv - recommended)
uv sync --extra dev

# Run the application
uv run streamlit run app.py

# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_datasets.py

# Run tests with coverage
uv run pytest --cov=skplay

# Lint code
uv run ruff check .

# Format code
uv run ruff format .

# Type checking
uv run mypy skplay
```

## Architecture

The codebase follows a layered architecture with clear separation between UI and core logic:

```
app.py                    Entry point and navigation
    ↓
pages/                    14 educational pages (Streamlit multipage)
    ↓
skplay/ui/                Reusable UI components and level system
    ↓
skplay/core/              Core ML engine (UI-independent)
    ↓
sklearn, pandas, numpy    Dependencies
```

### Core Engine (`skplay/core/`)

The core modules can be used independently of Streamlit:

- **datasets.py** - `DatasetRegistry` manages all datasets with metadata (`DatasetCard`). Supports sklearn built-in and synthetic domain-specific datasets.
- **estimators.py** - `EstimatorRegistry` organizes 30+ estimators by task type with level-based filtering.
- **preprocessing.py** - `PreprocessingBuilder` creates `ColumnTransformer` pipelines handling mixed-type data.
- **evaluation.py** - Metrics computation and visualizations for all task types.
- **tuning.py** - Hyperparameter tuning with GridSearchCV/RandomizedSearchCV wrappers.
- **snippets.py** - Generates reproducible Python code from trained pipelines.
- **upload.py** - CSV parsing with automatic dtype detection and task type inference.
- **api_explorer.py** - sklearn API introspection for the API Explorer page.

### UI Layer (`skplay/ui/`)

- **level.py** - Level system management (`Level` type, `level_selector()`, conditional parameter rendering).
- **components.py** - Reusable widgets: dataset cards, parameter controls, metrics tables, download buttons.

### Pages (`pages/`)

Each page follows the sklearn User Guide structure. Pages use tabs for workflow: Data → Preprocessing → Model → Results → Learn More.

## Key Patterns

**Registry Pattern**: Datasets and estimators use registries for extensibility. Adding new items requires:
1. Create the item (loader function or `EstimatorInfo`)
2. Register via `Registry.register()`
3. Item auto-appears in relevant pages

**Pipeline-First**: All models wrap in sklearn `Pipeline` (preprocessing + modeling) for consistency and code generation.

**Level-Based Filtering**: Same pages render different complexity based on `session_state["user_level"]`. Use `should_show_param()` for conditional parameter display.

**Caching**: Use `@st.cache_data` for dataset loading and `@st.cache_resource` for registry initialization.

## Adding New Content

**New Dataset**: Create loader returning `DatasetResult`, register in `datasets.py` with `DatasetCard` metadata.

**New Estimator**: Create `EstimatorInfo` in `estimators.py`, register with task type, add parameter grid in `tuning.py`.

**New Page**: Create `pages/NN_page_name.py`. Streamlit auto-discovers pages. Use `level_selector()` from `skplay.ui` and core engine for ML operations.

## Configuration

- **Python**: 3.10, 3.11, or 3.12
- **Line length**: 100 characters (ruff)
- **Test paths**: `tests/`
