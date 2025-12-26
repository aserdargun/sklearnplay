# Architecture

## Overview

sklearn-playground is a Streamlit multipage application structured around the sklearn User Guide. It provides an interactive learning experience with progressive complexity levels.

## Components

### Core Engine (`skplay/core/`)

The core engine is independent of the UI and can be reused:

#### `datasets.py`
- `DatasetRegistry`: Central registry for all datasets
- `DatasetResult`: Container for X, y, and metadata
- `DatasetCard`: Metadata about datasets (schema, task type, domain)
- Loaders for sklearn built-in and synthetic datasets

#### `upload.py`
- CSV parsing and validation
- Automatic dtype detection
- Target column selection
- Task type inference

#### `preprocessing.py`
- `PreprocessingBuilder`: Factory for preprocessing pipelines
- ColumnTransformer assembly for mixed types
- Imputation, scaling, encoding options
- Level-gated options

#### `estimators.py`
- `EstimatorRegistry`: Registry of estimators by task type
- `EstimatorInfo`: Metadata about estimators
- Level-based filtering (beginner/intermediate/advanced)
- Factory methods for creating estimators

#### `tuning.py`
- Parameter grids by estimator and level
- Grid search, random search, halving search wrappers
- Result formatting

#### `evaluation.py`
- Metric computation for all task types
- Cross-validation wrappers
- Visualization functions (ROC, confusion matrix, residuals, etc.)

#### `snippets.py`
- Code generation for reproducibility
- Pipeline and estimator serialization to code
- Import statement generation

#### `api_explorer.py`
- sklearn API introspection
- Search and filtering
- Signature and docstring extraction
- Example code generation

### UI Components (`skplay/ui/`)

#### `components.py`
- Reusable Streamlit widgets
- Dataset cards, metrics tables
- Parameter controls with type-aware widgets
- Download buttons, code display

#### `level.py`
- Level state management
- Level selector widgets
- Conditional rendering based on level
- Configuration by level

### Pages (`pages/`)

Each page follows the sklearn User Guide structure:
- Concept explanation
- Interactive demo using core engine
- "What to tune" panel
- Links to official docs

## Data Flow

```
User Input → UI Components → Core Engine → Evaluation → Results Display
     ↑                                                        ↓
     └──────────────────── Code Snippet ──────────────────────┘
```

1. User selects dataset, preprocessing, and model via UI
2. Core engine builds Pipeline with ColumnTransformer
3. Model is trained and evaluated
4. Results displayed with visualizations
5. Code snippet generated for reproducibility

## Level System

The level system controls UI complexity:

| Level | Features |
|-------|----------|
| Beginner | Essential params, guided flow, detailed tooltips |
| Intermediate | CV, regularization, learning curves |
| Advanced | Full params, search methods, export |

Levels affect:
- Which estimators are shown
- Which parameters are exposed
- Which visualizations are available
- Which preprocessing options appear

## State Management

Streamlit session state is used for:
- Current user level
- Selected dataset
- Preprocessing configuration
- Trained model
- Evaluation results

## Extension Points

### Adding Datasets
1. Create loader in `datasets.py`
2. Define `DatasetCard` with metadata
3. Register with `DatasetRegistry`

### Adding Estimators
1. Create `EstimatorInfo` in `estimators.py`
2. Register with `EstimatorRegistry`
3. Optionally add parameter grid in `tuning.py`

### Adding Pages
1. Create page file in `pages/`
2. Use level gating from `skplay.ui.level`
3. Use components from `skplay.ui.components`
4. Use core engine for ML operations

## Testing Strategy

- Unit tests for core modules
- Integration tests for pipelines
- No UI tests (rely on Streamlit's testing)

## Performance Considerations

- Dataset loading is cached with `@st.cache_data`
- Heavy computations show progress spinners
- Large operations use `n_jobs=-1` where possible
- API explorer lazy-loads on first access
