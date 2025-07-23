# feature_engineering.py
"""
Feature engineering module for customer churn prediction.
Creates, transforms, and selects features for machine learning models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
import yaml
import joblib
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom feature engineering transformer for churn prediction."""
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        """Initialize feature engineer with configuration."""
        self.config = self._load_config(config_path)
        self.scalers = {}
        self.feature_selectors = {}
        self.feature_names = []
        self.engineered_features = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load feature engineering configuration."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config.get('feature_engineering', {})
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default configuration.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default feature engineering configuration."""
        return {
            'scaling': {
                'method': 'standard',  # 'standard', 'minmax', 'robust'
                'features': ['tenure', 'MonthlyCharges', 'TotalCharges']
            },
            'polynomial_features': {
                'enabled': True,
                'degree': 2,
                'interaction_only': True,
                'features': ['tenure', 'MonthlyCharges']
            },
            'binning': {
                'enabled': True,
                'features': {
                    'tenure': {'bins': [0, 12, 24, 48, 72, 100], 'strategy': 'custom'},
                    'MonthlyCharges': {'bins': 4, 'strategy': 'quantile'}
                }
            },
            'feature_selection': {
                'enabled': True,
                'method': 'k_best',  # 'k_best', 'mutual_info', 'chi2'
                'k': 20
            },
            'dimensionality_reduction': {
                'enabled': False,
                'method': 'pca',
                'n_components': 0.95
            }
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'ChurnFeatureEngineer':
        """Fit feature engineering transformers."""
        logger.info("Fitting feature engineering transformers")
        
        X_copy = X.copy()
        self.feature_names = list(X_copy.columns)
        
        # Create basic engineered features
        X_engineered = self._create_engineered_features(X_copy)
        
        # Fit scalers
        X_scaled = self._fit_scalers(X_engineered, y)
        
        # Create polynomial features
        X_poly = self._create_polynomial_features(X_scaled)
        
        # Create binned features
        X_binned = self._create_binned_features(X_poly)
        
        # Fit feature selectors
        if self.config.get('feature_selection', {}).get('enabled', False):
            self._fit_feature_selectors(X_binned, y)
        
        # Fit dimensionality reduction
        if self.config.get('dimensionality_reduction', {}).get('enabled', False):
            self._fit_dimensionality_reduction(X_binned, y)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted transformers."""
        logger.info("Transforming features")
        
        X_copy = X.copy()
        
        # Create basic engineered features
        X_engineered = self._create_engineered_features(X_copy)
        
        # Apply scaling
        X_scaled = self._apply_scaling(X_engineered)
        
        # Create polynomial features
        X_poly = self._create_polynomial_features(X_scaled)
        
        # Create binned features
        X_binned = self._create_binned_features(X_poly)
        
        # Apply feature selection
        if self.config.get('feature_selection', {}).get('enabled', False):
            X_selected = self._apply_feature_selection(X_binned)
        else:
            X_selected = X_binned
        
        # Apply dimensionality reduction
        if self.config.get('dimensionality_reduction', {}).get('enabled', False):
            X_reduced = self._apply_dimensionality_reduction(X_selected)
        else:
            X_reduced = X_selected
        
        return X_reduced
    
    def _create_engineered_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features from raw data."""
        logger.info("Creating engineered features")
        
        # Customer Lifetime Value
        if 'tenure' in X.columns and 'MonthlyCharges' in X.columns:
            X['CLV'] = X['tenure'] * X['MonthlyCharges']
            self.engineered_features.append('CLV')
        
        # Average Monthly Charges
        if 'TotalCharges' in X.columns and 'tenure' in X.columns:
            X['AvgMonthlyCharges'] = X['TotalCharges'] / (X['tenure'] + 1)
            self.engineered_features.append('AvgMonthlyCharges')
        
        # Charge per tenure ratio
        if 'MonthlyCharges' in X.columns and 'tenure' in X.columns:
            X['ChargePerTenure'] = X['MonthlyCharges'] / (X['tenure'] + 1)
            self.engineered_features.append('ChargePerTenure')
        
        # Service count (count of services used)
        service_columns = [
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        available_services = [col for col in service_columns if col in X.columns]
        if available_services:
            X['ServiceCount'] = X[available_services].sum(axis=1)
            self.engineered_features.append('ServiceCount')
        
        # Premium service indicator
        premium_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
        available_premium = [col for col in premium_services if col in X.columns]
        if available_premium:
            X['PremiumServiceCount'] = X[available_premium].sum(axis=1)
            X['HasPremiumService'] = (X['PremiumServiceCount'] > 0).astype(int)
            self.engineered_features.extend(['PremiumServiceCount', 'HasPremiumService'])
        
        # Streaming service indicator
        streaming_services = ['StreamingTV', 'StreamingMovies']
        available_streaming = [col for col in streaming_services if col in X.columns]
        if available_streaming:
            X['StreamingCount'] = X[available_streaming].sum(axis=1)
            X['HasStreaming'] = (X['StreamingCount'] > 0).astype(int)
            self.engineered_features.extend(['StreamingCount', 'HasStreaming'])
        
        # Contract duration indicator
        if 'Contract' in X.columns:
            # Assuming Contract is encoded: 0=Month-to-month, 1=One year, 2=Two year
            X['IsLongTermContract'] = (X['Contract'] >= 1).astype(int)
            self.engineered_features.append('IsLongTermContract')
        
        # Senior citizen with high charges
        if 'SeniorCitizen' in X.columns and 'MonthlyCharges' in X.columns:
            X['SeniorHighCharges'] = ((X['SeniorCitizen'] == 1) & 
                                     (X['MonthlyCharges'] > X['MonthlyCharges'].median())).astype(int)
            self.engineered_features.append('SeniorHighCharges')
        
        # Total charges vs expected charges
        if 'TotalCharges' in X.columns and 'MonthlyCharges' in X.columns and 'tenure' in X.columns:
            X['ExpectedCharges'] = X['MonthlyCharges'] * X['tenure']
            X['ChargesRatio'] = X['TotalCharges'] / (X['ExpectedCharges'] + 1)
            self.engineered_features.extend(['ExpectedCharges', 'ChargesRatio'])
        
        logger.info(f"Created {len(self.engineered_features)} engineered features")
        return X
    
    def _fit_scalers(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit scalers for numerical features."""
        scaling_config = self.config.get('scaling', {})
        if not scaling_config.get('enabled', True):
            return X
        
        method = scaling_config.get('method', 'standard')
        features_to_scale = scaling_config.get('features', [])
        
        # Add engineered numerical features
        numerical_features = features_to_scale + [
            'CLV', 'AvgMonthlyCharges', 'ChargePerTenure', 'ExpectedCharges', 'ChargesRatio'
        ]
        
        # Filter features that exist in the data
        available_features = [col for col in numerical_features if col in X.columns]
        
        if not available_features:
            return X
        
        # Choose scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaling method: {method}. Using standard scaler.")
            scaler = StandardScaler()
        
        # Fit scaler
        scaler.fit(X[available_features])
        self.scalers['numerical'] = scaler
        
        logger.info(f"Fitted {method} scaler for {len(available_features)} features")
        return X
    
    def _apply_scaling(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scalers to features."""
        if 'numerical' not in self.scalers:
            return X
        
        X_scaled = X.copy()
        scaler = self.scalers['numerical']
        
        # Get features that were scaled during fit
        features_to_scale = list(scaler.feature_names_in_)
        available_features = [col for col in features_to_scale if col in X_scaled.columns]
        
        if available_features:
            X_scaled[available_features] = scaler.transform(X_scaled[available_features])
        
        return X_scaled
    
    def _create_polynomial_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features."""
        poly_config = self.config.get('polynomial_features', {})
        if not poly_config.get('enabled', False):
            return X
        
        from sklearn.preprocessing import PolynomialFeatures
        
        features_for_poly = poly_config.get('features', [])
        available_features = [col for col in features_for_poly if col in X.columns]
        
        if not available_features:
            return X
        
        # Create polynomial features
        poly = PolynomialFeatures(
            degree=poly_config.get('degree', 2),
            interaction_only=poly_config.get('interaction_only', True),
            include_bias=False
        )
        
        poly_features = poly.fit_transform(X[available_features])
        poly_feature_names = [f"poly_{name}" for name in poly.get_feature_names_out(available_features)]
        
        # Add polynomial features to dataframe
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=X.index)
        X_poly = pd.concat([X, poly_df], axis=1)
        
        # Store polynomial transformer
        self.scalers['polynomial'] = poly
        
        logger.info(f"Created {len(poly_feature_names)} polynomial features")
        return X_poly
    
    def _create_binned_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create binned features."""
        binning_config = self.config.get('binning', {})
        if not binning_config.get('enabled', False):
            return X
        
        X_binned = X.copy()
        
        for feature, bin_config in binning_config.get('features', {}).items():
            if feature not in X_binned.columns:
                continue
            
            strategy = bin_config.get('strategy', 'quantile')
            bins = bin_config.get('bins', 4)
            
            if strategy == 'custom' and isinstance(bins, list):
                # Custom bins
                X_binned[f'{feature}_binned'] = pd.cut(
                    X_binned[feature], 
                    bins=bins, 
                    labels=False, 
                    include_lowest=True
                )
            elif strategy == 'quantile':
                # Quantile-based bins
                X_binned[f'{feature}_binned'] = pd.qcut(
                    X_binned[feature], 
                    q=bins, 
                    labels=False, 
                    duplicates='drop'
                )
            else:
                # Equal-width bins
                X_binned[f'{feature}_binned'] = pd.cut(
                    X_binned[feature], 
                    bins=bins, 
                    labels=False, 
                    include_lowest=True
                )
        
        return X_binned
    
    def _fit_feature_selectors(self, X: pd.DataFrame, y: pd.Series):
        """Fit feature selectors."""
        selection_config = self.config.get('feature_selection', {})
        method = selection_config.get('method', 'k_best')
        k = selection_config.get('k', 20)
        
        # Choose selector
        if method == 'k_best':
            selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif method == 'chi2':
            selector = SelectKBest(score_func=chi2, k=k)
        else:
            logger.warning(f"Unknown feature selection method: {method}. Using k_best.")
            selector = SelectKBest(score_func=f_classif, k=k)
        
        # Fit selector
        selector.fit(X, y)
        self.feature_selectors['main'] = selector
        
        logger.info(f"Fitted feature selector: {method} with k={k}")
    
    def _apply_feature_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection."""
        if 'main' not in self.feature_selectors:
            return X
        
        # Drop rows with NaN values before feature selection
        X_clean = X.dropna()
        
        selector = self.feature_selectors['main']
        X_selected = selector.transform(X_clean)
        
        # Get selected feature names
        selected_features = X_clean.columns[selector.get_support()].tolist()
        
        # Create dataframe with selected features
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X_clean.index)
        
        logger.info(f"Selected {len(selected_features)} features from {len(X_clean.columns)}")
        return X_selected_df
    
    def _fit_dimensionality_reduction(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit dimensionality reduction."""
        reduction_config = self.config.get('dimensionality_reduction', {})
        method = reduction_config.get('method', 'pca')
        n_components = reduction_config.get('n_components', 0.95)
        
        if method == 'pca':
            reducer = PCA(n_components=n_components)
        else:
            logger.warning(f"Unknown dimensionality reduction method: {method}. Using PCA.")
            reducer = PCA(n_components=n_components)
        
        # Fit reducer
        reducer.fit(X)
        self.scalers['dimensionality_reduction'] = reducer
        
        logger.info(f"Fitted {method} with n_components={n_components}")
    
    def _apply_dimensionality_reduction(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply dimensionality reduction."""
        if 'dimensionality_reduction' not in self.scalers:
            return X
        
        reducer = self.scalers['dimensionality_reduction']
        X_reduced = reducer.transform(X)
        
        # Create column names
        n_components = X_reduced.shape[1]
        column_names = [f'PC_{i+1}' for i in range(n_components)]
        
        # Create dataframe
        X_reduced_df = pd.DataFrame(X_reduced, columns=column_names, index=X.index)
        
        logger.info(f"Reduced dimensions from {X.shape[1]} to {n_components}")
        return X_reduced_df
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from feature selectors."""
        if 'main' not in self.feature_selectors:
            return {}
        
        selector = self.feature_selectors['main']
        feature_names = self.feature_names + self.engineered_features
        
        # Get scores
        scores = selector.scores_
        
        # Create importance dictionary
        importance = {}
        for i, name in enumerate(feature_names):
            if i < len(scores):
                importance[name] = scores[i]
        
        # Sort by importance
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def save_transformers(self, output_path: str = "models/transformers/"):
        """Save fitted transformers."""
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # Save feature engineer
        joblib.dump(self, Path(output_path) / "feature_engineer.joblib")
        
        # Save individual transformers
        for name, transformer in self.scalers.items():
            joblib.dump(transformer, Path(output_path) / f"{name}_transformer.joblib")
        
        for name, selector in self.feature_selectors.items():
            joblib.dump(selector, Path(output_path) / f"{name}_selector.joblib")
        
        logger.info(f"Transformers saved to {output_path}")
    
    def load_transformers(self, input_path: str = "models/transformers/"):
        """Load fitted transformers."""
        transformer_path = Path(input_path)
        
        # Load scalers
        for transformer_file in transformer_path.glob("*_transformer.joblib"):
            name = transformer_file.stem.replace("_transformer", "")
            transformer = joblib.load(transformer_file)
            self.scalers[name] = transformer
        
        # Load selectors
        for selector_file in transformer_path.glob("*_selector.joblib"):
            name = selector_file.stem.replace("_selector", "")
            selector = joblib.load(selector_file)
            self.feature_selectors[name] = selector
        
        logger.info(f"Transformers loaded from {input_path}")

def main():
    """Main function for testing feature engineering."""
    # Example usage
    from src.data.data_loader import DataLoader
    
    try:
        # Load data
        loader = DataLoader()
        train_df = loader.load_processed_data("train")
        
        # Split features and target
        X_train, y_train = loader.get_feature_target_split(train_df)
        
        # Create feature engineer
        feature_engineer = ChurnFeatureEngineer()
        
        # Fit and transform
        feature_engineer.fit(X_train, y_train)
        X_transformed = feature_engineer.transform(X_train)
        
        # Print results
        print(f"Original features: {X_train.shape[1]}")
        print(f"Transformed features: {X_transformed.shape[1]}")
        print(f"Feature names: {list(X_transformed.columns)}")
        
        # Get feature importance
        importance = feature_engineer.get_feature_importance()
        print(f"\nTop 10 most important features:")
        for feature, score in list(importance.items())[:10]:
            print(f"  {feature}: {score:.4f}")
        
        # Save transformers
        feature_engineer.save_transformers()
        
    except Exception as e:
        print(f"Error in feature engineering: {str(e)}")

if __name__ == "__main__":
    main()