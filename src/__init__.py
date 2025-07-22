"""
src package - Core implementation of the MLOps pipeline
"""

# Expose key modules at package level
from .data.data_ingestion import DataIngestion
from .utils.logger import get_logger

__all__ = ["DataIngestion", "get_logger"]  # For `from src import *`