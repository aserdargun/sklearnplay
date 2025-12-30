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
from skplay.ui.styles import (
    dataset_row,
    feature_card,
    glass_card,
    glass_divider,
    hero_section,
    inject_styles,
    nav_card,
    section_header,
)

__all__ = [
    # Components
    "show_dataset_card",
    "show_metrics_table",
    "show_parameter_controls",
    "show_code_snippet",
    "create_download_button",
    # Level system
    "get_level",
    "set_level",
    "level_selector",
    "show_for_level",
    "get_level_config",
    # Styles
    "inject_styles",
    "hero_section",
    "glass_card",
    "nav_card",
    "feature_card",
    "section_header",
    "glass_divider",
    "dataset_row",
]
