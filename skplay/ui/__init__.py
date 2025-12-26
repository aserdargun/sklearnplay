"""UI components and helpers for the Streamlit app."""

from skplay.ui.components import (
    create_download_button,
    show_code_snippet,
    show_dataset_card,
    show_metrics_table,
    show_parameter_controls,
)
from skplay.ui.level import (
    get_level,
    get_level_config,
    level_selector,
    set_level,
    show_for_level,
)

__all__ = [
    "show_dataset_card",
    "show_metrics_table",
    "show_parameter_controls",
    "show_code_snippet",
    "create_download_button",
    "get_level",
    "set_level",
    "level_selector",
    "show_for_level",
    "get_level_config",
]
