"""User level management and gating.

Provides level selection, configuration, and conditional UI rendering.
"""

from collections.abc import Callable
from functools import wraps
from typing import Any, Literal, cast

import streamlit as st

Level = Literal["beginner", "intermediate", "advanced"]

LEVEL_ORDER = {"beginner": 0, "intermediate": 1, "advanced": 2}

LEVEL_DESCRIPTIONS = {
    "beginner": "Guided experience with essential controls and clear explanations",
    "intermediate": "More options for model tuning and validation",
    "advanced": "Full control with all parameters and advanced features",
}

LEVEL_ICONS = {
    "beginner": "🌱",
    "intermediate": "🌿",
    "advanced": "🌳",
}


def get_level() -> Level:
    """Get the current user level from session state.

    Returns:
        Current level (defaults to 'beginner')
    """
    if "user_level" not in st.session_state:
        st.session_state.user_level = "beginner"
    return cast(Level, st.session_state.user_level)


def set_level(level: Level) -> None:
    """Set the user level in session state.

    Args:
        level: The level to set
    """
    st.session_state.user_level = level


def level_selector(key: str = "level_selector") -> Level:
    """Render level selector widget.

    Args:
        key: Unique key for the widget

    Returns:
        Selected level
    """
    current = get_level()

    options = list(LEVEL_ORDER.keys())
    index = options.index(current)

    selected = st.radio(
        "Experience Level",
        options=options,
        index=index,
        format_func=lambda x: f"{LEVEL_ICONS[x]} {x.title()}",
        help="Choose your experience level to adjust the interface complexity",
        key=key,
        horizontal=True,
    )

    if selected != current:
        set_level(cast(Level, selected))

    return cast(Level, selected)


def level_selector_sidebar() -> Level:
    """Render level selector in sidebar.

    Returns:
        Selected level
    """
    with st.sidebar:
        st.subheader("Experience Level")

        current = get_level()
        options = list(LEVEL_ORDER.keys())

        for level in options:
            if st.button(
                f"{LEVEL_ICONS[level]} {level.title()}",
                key=f"level_btn_{level}",
                type="primary" if level == current else "secondary",
                width="stretch",
            ):
                set_level(cast(Level, level))
                st.rerun()

        # Show current level description
        st.caption(LEVEL_DESCRIPTIONS[current])

    return get_level()


def show_for_level(
    min_level: Level,
    content_fn: Callable[[], Any],
    placeholder_text: str | None = None,
) -> Any | None:
    """Conditionally show content based on user level.

    Args:
        min_level: Minimum level required to show content
        content_fn: Function that renders the content
        placeholder_text: Optional text to show if level not met

    Returns:
        Result of content_fn if shown, None otherwise
    """
    current = get_level()
    current_order = LEVEL_ORDER[current]
    required_order = LEVEL_ORDER[min_level]

    if current_order >= required_order:
        return content_fn()
    elif placeholder_text:
        st.info(f"🔒 {placeholder_text} (requires {min_level} level)")
    return None


def level_gate(min_level: Level):
    """Decorator to gate a function by user level.

    Args:
        min_level: Minimum level required

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current = get_level()
            if LEVEL_ORDER[current] >= LEVEL_ORDER[min_level]:
                return func(*args, **kwargs)
            return None

        return wrapper

    return decorator


def get_level_config() -> dict[str, Any]:
    """Get configuration based on current level.

    Returns:
        Configuration dictionary
    """
    level = get_level()

    configs = {
        "beginner": {
            "show_cv": False,
            "show_feature_selection": False,
            "show_dim_reduction": False,
            "show_learning_curve": False,
            "show_validation_curve": False,
            "show_permutation_importance": False,
            "show_pdp": False,
            "show_code_snippet": False,
            "show_model_export": False,
            "show_grid_search": False,
            "show_random_search": False,
            "show_halving_search": False,
            "max_params_shown": 4,
            "param_categories": ["essential"],
            "tooltip_verbosity": "high",
            "default_cv_folds": 5,
            "show_advanced_metrics": False,
        },
        "intermediate": {
            "show_cv": True,
            "show_feature_selection": True,
            "show_dim_reduction": False,
            "show_learning_curve": True,
            "show_validation_curve": True,
            "show_permutation_importance": True,
            "show_pdp": False,
            "show_code_snippet": True,
            "show_model_export": True,
            "show_grid_search": True,
            "show_random_search": True,
            "show_halving_search": False,
            "max_params_shown": 10,
            "param_categories": ["essential", "regularization", "performance"],
            "tooltip_verbosity": "medium",
            "default_cv_folds": 5,
            "show_advanced_metrics": True,
        },
        "advanced": {
            "show_cv": True,
            "show_feature_selection": True,
            "show_dim_reduction": True,
            "show_learning_curve": True,
            "show_validation_curve": True,
            "show_permutation_importance": True,
            "show_pdp": True,
            "show_code_snippet": True,
            "show_model_export": True,
            "show_grid_search": True,
            "show_random_search": True,
            "show_halving_search": True,
            "max_params_shown": 100,
            "param_categories": ["all"],
            "tooltip_verbosity": "low",
            "default_cv_folds": 5,
            "show_advanced_metrics": True,
        },
    }

    return configs[level]


def get_param_visibility(
    param_name: str,
    estimator_name: str,
) -> Level:
    """Determine minimum level to show a parameter.

    Args:
        param_name: Parameter name
        estimator_name: Estimator name

    Returns:
        Minimum level required
    """
    # Essential params shown to all
    essential_params = {
        "n_estimators",
        "max_depth",
        "n_neighbors",
        "C",
        "alpha",
        "n_clusters",
        "kernel",
        "contamination",
        "learning_rate",
    }

    # Intermediate-level params
    intermediate_params = {
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "penalty",
        "solver",
        "gamma",
        "eps",
        "min_samples",
        "subsample",
        "reg_alpha",
        "reg_lambda",
        "l2_regularization",
        "criterion",
        "splitter",
        "init",
        "n_init",
        "max_iter",
        "tol",
        "warm_start",
        "class_weight",
        "n_jobs",
    }

    if param_name in essential_params:
        return "beginner"
    elif param_name in intermediate_params:
        return "intermediate"
    else:
        return "advanced"


def should_show_param(param_name: str, estimator_name: str) -> bool:
    """Check if a parameter should be shown at current level.

    Args:
        param_name: Parameter name
        estimator_name: Estimator name

    Returns:
        True if parameter should be shown
    """
    current = get_level()
    required = get_param_visibility(param_name, estimator_name)
    return LEVEL_ORDER[current] >= LEVEL_ORDER[required]


def get_explanation_depth() -> Literal["brief", "moderate", "detailed"]:
    """Get explanation depth based on level.

    Returns:
        Explanation depth
    """
    level = get_level()
    depth_map: dict[Level, Literal["brief", "moderate", "detailed"]] = {
        "beginner": "detailed",
        "intermediate": "moderate",
        "advanced": "brief",
    }
    return depth_map[level]


def format_help_text(
    brief: str,
    detailed: str | None = None,
) -> str:
    """Format help text based on user level.

    Args:
        brief: Brief explanation
        detailed: Detailed explanation (shown for beginners)

    Returns:
        Formatted help text
    """
    depth = get_explanation_depth()

    if depth == "detailed" and detailed:
        return f"{brief}\n\n{detailed}"
    return brief
