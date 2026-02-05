# Configuration file for the Sphinx documentation builder.

project = 'MLOps Churn Prediction'
copyright = '2024, MLOps Team'
author = 'MLOps Team'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

source_suffix = {
    '.rst': None,
    '.md': None,
}