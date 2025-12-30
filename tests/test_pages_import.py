"""Tests that all pages can be imported without errors.

These tests ensure there are no syntax errors or import issues in the page files.
Note: These don't test the full Streamlit rendering, just the module imports.
"""

import importlib

import pytest

# List of all page modules
PAGE_MODULES = [
    "pages.01_supervised_learning",
    "pages.02_unsupervised_learning",
    "pages.03_model_selection",
    "pages.04_metadata_routing",
    "pages.05_inspection",
    "pages.06_visualizations",
    "pages.07_transformations",
    "pages.08_dataset_loading",
    "pages.09_computing",
    "pages.10_persistence",
    "pages.11_pitfalls",
    "pages.12_dispatching",
    "pages.13_choosing_estimator",
    "pages.14_external_resources",
    "pages.90_api_explorer",
]


class TestPageImports:
    """Tests for page module imports."""

    @pytest.mark.parametrize("module_name", PAGE_MODULES)
    def test_page_imports(self, module_name):
        """Test that each page module can be imported without errors.

        This catches:
        - Syntax errors
        - Missing imports
        - Module-level exceptions

        Note: Streamlit-specific code won't execute without a Streamlit context,
        but the imports and module-level definitions should work.
        """
        try:
            # Import the module
            module = importlib.import_module(module_name)
            assert module is not None
        except ImportError as e:
            pytest.fail(f"Import error in {module_name}: {e}")
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {module_name}: {e}")
        except Exception as e:
            # Some pages may have code that runs on import that requires
            # Streamlit context - this is expected for Streamlit apps
            # We check for specific error types that indicate real problems
            error_str = str(e)
            # Allow Streamlit-related errors since we're not running in Streamlit
            if "streamlit" in error_str.lower() or "st." in error_str:
                pass  # Expected error without Streamlit context
            else:
                pytest.fail(f"Unexpected error in {module_name}: {e}")


class TestAppImport:
    """Tests for main app import."""

    def test_app_imports(self):
        """Test that the main app can be imported."""
        try:
            import app
            assert app is not None
        except ImportError as e:
            pytest.fail(f"Import error in app: {e}")
        except SyntaxError as e:
            pytest.fail(f"Syntax error in app: {e}")


class TestSkplayModules:
    """Tests for skplay package imports."""

    def test_skplay_core_imports(self):
        """Test that all core modules can be imported."""
        modules = [
            "skplay.core.datasets",
            "skplay.core.estimators",
            "skplay.core.preprocessing",
            "skplay.core.evaluation",
            "skplay.core.tuning",
            "skplay.core.snippets",
            "skplay.core.upload",
            "skplay.core.api_explorer",
        ]

        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None
            except Exception as e:
                pytest.fail(f"Error importing {module_name}: {e}")

    def test_skplay_ui_imports(self):
        """Test that UI modules can be imported."""
        modules = [
            "skplay.ui.level",
            "skplay.ui.components",
            "skplay.ui.styles",
        ]

        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None
            except Exception as e:
                # UI modules may require Streamlit context
                error_str = str(e)
                if "streamlit" not in error_str.lower():
                    pytest.fail(f"Error importing {module_name}: {e}")

    def test_skplay_backend_imports(self):
        """Test that backend modules can be imported.

        Note: Backend modules require optional 'azure' dependencies (sqlalchemy, etc.)
        If these aren't installed, the test is skipped.
        """
        # Check if sqlalchemy is available (required for backend)
        import importlib.util
        if importlib.util.find_spec("sqlalchemy") is None:
            pytest.skip("Backend modules require optional 'azure' dependencies (sqlalchemy)")

        modules = [
            "skplay.backend.storage",
            "skplay.backend.experiments",
        ]

        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None
            except Exception as e:
                pytest.fail(f"Error importing {module_name}: {e}")
