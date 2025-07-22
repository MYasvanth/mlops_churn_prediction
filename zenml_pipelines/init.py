# init.py
"""
ZenML Pipeline Definitions Package

This package contains all ZenML pipeline definitions for the churn prediction MLOps project.
"""

__version__ = "0.1.0"
__author__ = "MLOps Team"

# Import main pipelines for easy access
from .training_pipeline import training_pipeline
from .inference_pipeline import inference_pipeline
from .monitoring_pipeline import monitoring_pipeline

__all__ = [
    "training_pipeline",
    "inference_pipeline", 
    "monitoring_pipeline"
]