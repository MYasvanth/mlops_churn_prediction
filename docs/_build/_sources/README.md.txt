# Documentation

This directory contains comprehensive documentation for the MLOps Churn Prediction project.

## Structure

```
docs/
├── index.rst              # Main documentation index
├── installation.rst       # Installation guide
├── usage.rst              # Usage examples
├── api_documentation.md   # API reference
├── deployment_guide.md    # Deployment instructions
├── pipeline_documentation.md # Pipeline details
├── monitoring.rst         # Monitoring setup
├── conf.py               # Sphinx configuration
├── Makefile              # Build commands
├── build_docs.py         # Documentation build script
└── requirements.txt      # Documentation dependencies
```

## Building Documentation

### Quick Build

```bash
python build_docs.py
```

### Manual Build

```bash
# Install dependencies
pip install -r requirements.txt

# Build HTML
make html

# View documentation
open _build/html/index.html
```

## Documentation Types

1. **Installation Guide** (`installation.rst`) - Setup instructions
2. **Usage Guide** (`usage.rst`) - Practical examples
3. **API Documentation** (`api_documentation.md`) - REST API reference
4. **Deployment Guide** (`deployment_guide.md`) - Production deployment
5. **Pipeline Documentation** (`pipeline_documentation.md`) - ML pipeline details
6. **Monitoring Guide** (`monitoring.rst`) - Monitoring and alerting

## Access Points

- **Built Documentation**: `_build/html/index.html`
- **API Docs**: http://localhost:8000/docs (when API is running)
- **Streamlit Dashboard**: http://localhost:8501
- **MLflow UI**: http://localhost:5000

## Contributing

When adding new documentation:

1. Follow existing structure and formatting
2. Update `index.rst` table of contents
3. Rebuild documentation to verify changes
4. Test all links and examples