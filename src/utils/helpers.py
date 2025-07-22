"""
Helper utilities for the Customer Churn MLOps project.
Contains common utility functions used across different modules.
"""

import os
import json
import pickle
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_directories(paths: List[str]) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        paths: List of directory paths to create
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save the JSON file
    """
    create_directories([os.path.dirname(filepath)])
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved JSON data to: {filepath}")


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Dictionary loaded from JSON
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded JSON data from: {filepath}")
    return data


def save_pickle(obj: Any, filepath: str) -> None:
    """
    Save object to pickle file.
    
    Args:
        obj: Object to save
        filepath: Path to save the pickle file
    """
    create_directories([os.path.dirname(filepath)])
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    
    logger.info(f"Saved pickle object to: {filepath}")


def load_pickle(filepath: str) -> Any:
    """
    Load object from pickle file.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Object loaded from pickle
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pickle file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    
    logger.info(f"Loaded pickle object from: {filepath}")
    return obj


def save_joblib(obj: Any, filepath: str) -> None:
    """
    Save object using joblib (efficient for sklearn models).
    
    Args:
        obj: Object to save
        filepath: Path to save the file
    """
    create_directories([os.path.dirname(filepath)])
    
    joblib.dump(obj, filepath)
    logger.info(f"Saved joblib object to: {filepath}")


def load_joblib(filepath: str) -> Any:
    """
    Load object using joblib.
    
    Args:
        filepath: Path to the file
        
    Returns:
        Object loaded from file
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Joblib file not found: {filepath}")
    
    obj = joblib.load(filepath)
    logger.info(f"Loaded joblib object from: {filepath}")
    return obj


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary containing various classification metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
    }
    
    # Add ROC-AUC if probabilities are provided
    if y_pred_proba is not None:
        if len(np.unique(y_true)) == 2:  # Binary classification
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:  # Multi-class classification
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
    
    logger.info(f"Calculated classification metrics: {metrics}")
    return metrics


def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.
    
    Args:
        df: Input dataframe
        target_column: Name of the target column
        test_size: Size of the test set
        random_state: Random state for reproducibility
        stratify: Whether to stratify the split
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    stratify_column = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_column
    )
    
    logger.info(f"Split data: Train size={len(X_train)}, Test size={len(X_test)}")
    return X_train, X_test, y_train, y_test


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that dataframe contains required columns.
    
    Args:
        df: Input dataframe
        required_columns: List of required column names
        
    Returns:
        True if validation passes, False otherwise
    """
    missing_columns = set(required_columns) - set(df.columns)
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    logger.info("Dataframe validation passed")
    return True


def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive information about the dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary containing data information
    """
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'duplicates': df.duplicated().sum(),
    }
    
    # Add statistics for numerical columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_columns:
        info['numerical_stats'] = df[numerical_columns].describe().to_dict()
    
    # Add value counts for categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_columns:
        info['categorical_stats'] = {}
        for col in categorical_columns:
            info['categorical_stats'][col] = df[col].value_counts().to_dict()
    
    logger.info(f"Generated data info for dataframe with shape {df.shape}")
    return info


def log_data_quality_report(df: pd.DataFrame, stage: str = "unknown") -> None:
    """
    Log a comprehensive data quality report.
    
    Args:
        df: Input dataframe
        stage: Stage of the pipeline (e.g., "raw", "processed")
    """
    logger.info(f"=== Data Quality Report - {stage.upper()} ===")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info(f"Duplicates: {df.duplicated().sum()}")
    
    # Null values report
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        logger.warning("Null values found:")
        for col, count in null_counts[null_counts > 0].items():
            percentage = (count / len(df)) * 100
            logger.warning(f"  {col}: {count} ({percentage:.2f}%)")
    else:
        logger.info("No null values found")
    
    # Data types report
    logger.info(f"Data types: {df.dtypes.value_counts().to_dict()}")
    logger.info("=" * 50)


def encode_target_variable(target_series: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Encode target variable for binary classification.
    
    Args:
        target_series: Target variable series
        
    Returns:
        Tuple of (encoded_series, encoding_mapping)
    """
    from sklearn.preprocessing import LabelEncoder
    
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(target_series)
    
    # Create mapping dictionary
    mapping = {label: code for code, label in enumerate(encoder.classes_)}
    
    logger.info(f"Encoded target variable: {mapping}")
    return pd.Series(encoded, index=target_series.index), mapping


def get_feature_importance(model: Any, feature_names: List[str]) -> Dict[str, float]:
    """
    Extract feature importance from trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    if hasattr(model, 'feature_importances_'):
        importance_scores = model.feature_importances_
        feature_importance = dict(zip(feature_names, importance_scores))
        
        # Sort by importance
        feature_importance = dict(sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        logger.info(f"Extracted feature importance for {len(feature_names)} features")
        return feature_importance
    else:
        logger.warning("Model does not have feature_importances_ attribute")
        return {}


def convert_to_serializable(obj: Any) -> Any:
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        Serializable object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj