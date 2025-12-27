# StandardE2E Documentation

This directory contains the Sphinx-based documentation for StandardE2E.

## Building the Documentation

### Prerequisites

Install documentation dependencies:

```bash
pip install -r requirements.txt
```

### Building HTML Documentation

```bash
cd docs
make html
```

The built documentation will be in `_build/html/`. Open `_build/html/index.html` in your browser.

### Clean Build

To start fresh:

```bash
make clean
make html
```

## Documentation Structure

- **index.rst** - Main landing page with overview and navigation
- **quickstart.rst** - Quick start guide with interactive tutorials
- **overview.rst** - Architecture and key concepts  
- **user_guide.rst** - Comprehensive usage guide
- **reference/api.rst** - Auto-generated API documentation

## Interactive Tutorials

The documentation links to Jupyter notebooks in the `notebooks/` directory:

- `intro_tutorial.ipynb` - Introduction to StandardE2E
- `containers.ipynb` - Data containers deep dive
- `multi_dataset_training_and_filtering.ipynb` - Multi-dataset training
- `creating_custom_adapter.ipynb` - Custom adapters guide

## Contributing to Documentation

When adding new features, please update the relevant documentation:

1. Add docstrings to new classes/functions (auto-generated in API reference)
2. Update user_guide.rst for new workflows or configuration options
3. Update quickstart.rst if it affects the getting started experience
4. Add examples to notebooks/ for complex features

## Local Development

For faster iteration during documentation writing:

```bash
# Install sphinx-autobuild
pip install sphinx-autobuild

# Auto-rebuild on changes
sphinx-autobuild . _build/html
```

Then open http://127.0.0.1:8000 in your browser. The page will auto-reload on changes.
