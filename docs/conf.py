from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _get_version() -> str:
    try:
        from standard_e2e import __version__

        return __version__
    except Exception:
        return "0.0.0"


project = "StandardE2E"
author = "Stepan Konev and contributors"
release = _get_version()
version = release
copyright = f"{datetime.now().year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "myst_parser",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autosummary_generate = True
autodoc_typehints = "description"


# Exclude common inherited members from enums to reduce clutter
def autodoc_skip_member(app, what, name, obj, skip, options):
    """Skip inherited enum methods and Pydantic internals that clutter the documentation."""
    # List of inherited enum methods to exclude
    exclude_methods = {
        '__class__',
        '__delattr__',
        '__dir__',
        '__eq__',
        '__format__',
        '__ge__',
        '__getattribute__',
        '__gt__',
        '__hash__',
        '__init__',
        '__init_subclass__',
        '__le__',
        '__lt__',
        '__ne__',
        '__new__',
        '__reduce__',
        '__reduce_ex__',
        '__repr__',
        '__setattr__',
        '__sizeof__',
        '__str__',
        '__subclasshook__',
        '_generate_next_value_',
        '_member_names_',
        '_member_map_',
        '_member_type_',
        '_value2member_map_',
        '__doc__',
        '__module__',
    }

    # Pydantic internals to exclude
    pydantic_internals = {
        # Pydantic v2 internals
        'model_parametrized_name',
        '__pydantic_fields_set__',
        '__pydantic_core_schema__',
        '__pydantic_custom_init__',
        '__pydantic_decorators__',
        '__pydantic_generic_metadata__',
        '__pydantic_parent_namespace__',
        '__pydantic_post_init__',
        '__pydantic_private__',
        '__pydantic_serializer__',
        '__pydantic_validator__',
        '__pydantic_complete__',
        '__pydantic_fields__',
        '__pydantic_init_subclass__',
        'model_computed_fields',
        'model_config',
        'model_fields',
        'model_construct',
        'model_copy',
        'model_dump',
        'model_dump_json',
        'model_extra',
        'model_fields_set',
        'model_json_schema',
        'model_parametrized_name',
        'model_post_init',
        'model_rebuild',
        'model_validate',
        'model_validate_json',
        'model_validate_strings',
        '__get_pydantic_core_schema__',
        '__get_pydantic_json_schema__',
        # Pydantic v1 methods
        'construct',
        'copy',
        'dict',
        'from_orm',
        'json',
        'parse_file',
        'parse_obj',
        'parse_raw',
        'schema',
        'schema_json',
        'update_forward_refs',
        'validate',
        # Pydantic utility methods
        '__class_getitem__',
        '__copy__',
        '__deepcopy__',
        '__getstate__',
        '__setstate__',
        '__iter__',
        '__pretty__',
        '__replace__',
        '__repr_args__',
        '__repr_name__',
        '__repr_recursion__',
        '__repr_str__',
        '__rich_repr__',
        '_calculate_keys',
        '_check_frozen',
        '_copy_and_set_values',
        '_get_value',
        '_iter',
        '__getattr__',
        # Pydantic attributes
        '__dict__',
        '__pydantic_extra__',
        '__abstractmethods__',
        '__annotations__',
        '__class_vars__',
        '__fields_set__',
        '__private_attributes__',
        '__pydantic_computed_fields__',
        '__pydantic_root_model__',
        '__signature__',
        '__slots__',
        '__weakref__',
        '_abc_impl',
    }

    # Exclude private validation methods (starts with _validate_)
    if name.startswith('_validate_'):
        return True

    if name in exclude_methods or name in pydantic_internals:
        return True
    return skip


autodoc_mock_imports = [
    "torch",
    "tensorflow",
    "tensorflow_cpu",
    "tensorflow-cpu",
    "albumentations",
]
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

todo_include_todos = True

html_theme = "furo"
html_static_path = ["_static"]
html_logo = str(ROOT / "assets" / "standard_e2e_logo_contrast.png")
html_title = f"StandardE2E {release}"
html_favicon = str(ROOT / "assets" / "favicon.png")

html_theme_options = {
    "sidebar_hide_name": False,
    "light_css_variables": {
        "color-brand-primary": "#2563eb",
        "color-brand-content": "#1d4ed8",
    },
    "dark_css_variables": {
        "color-brand-primary": "#60a5fa",
        "color-brand-content": "#93c5fd",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/stepankonev/StandardE2E",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

# MyST parser configuration for better markdown support
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "substitution",
    "tasklist",
]


def setup(app):
    """Sphinx setup hook."""
    app.connect('autodoc-skip-member', autodoc_skip_member)
