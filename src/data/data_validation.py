"""
Data validation module for customer churn prediction.
Validates data quality, schema, and detects anomalies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, float]

class DataValidator:
    """Data validation class for customer churn dataset."""
    
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """Initialize data validator with configuration."""
        self.config = self._load_config(config_path)
        self.expected_columns = self.config.get('expected_columns', [])
        self.validation_rules = self.config.get('validation_rules', {})
        
    def _load_config(self, config_path: str) -> Dict:
        """Load validation configuration."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default validation rules.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default validation configuration for churn dataset."""
        return {
            'expected_columns': [
                'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
            ],
            'validation_rules': {
                'tenure': {'min': 0, 'max': 100},
                'MonthlyCharges': {'min': 0, 'max': 200},
                'TotalCharges': {'min': 0, 'max': 10000},
                'categorical_columns': [
                    'gender', 'Partner', 'Dependents', 'PhoneService',
                    'MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies', 'Contract',
                    'PaperlessBilling', 'PaymentMethod', 'Churn'
                ],
                'binary_columns': [
                    'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                    'PaperlessBilling', 'Churn'
                ],
                'required_columns': [
                    'customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn'
                ]
            }
        }
    
    def validate_schema(self, df: pd.DataFrame) -> ValidationResult:
        """Validate data schema."""
        errors = []
        warnings = []
        
        # Check if all expected columns are present
        missing_columns = set(self.expected_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing columns: {missing_columns}")
        
        # Check for unexpected columns
        unexpected_columns = set(df.columns) - set(self.expected_columns)
        if unexpected_columns:
            warnings.append(f"Unexpected columns: {unexpected_columns}")
        
        # Check data types
        type_errors = self._validate_data_types(df)
        errors.extend(type_errors)
        
        is_valid = len(errors) == 0
        metrics = {
            'total_columns': len(df.columns),
            'missing_columns': len(missing_columns),
            'unexpected_columns': len(unexpected_columns)
        }
        
        return ValidationResult(is_valid, errors, warnings, metrics)
    
    def _validate_data_types(self, df: pd.DataFrame) -> List[str]:
        """Validate data types for specific columns."""
        errors = []
        
        # Check numeric columns
        numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
        for col in numeric_columns:
            if col in df.columns:
                try:
                    pd.to_numeric(df[col], errors='coerce')
                except:
                    errors.append(f"Column {col} should be numeric")
        
        return errors
    
    def validate_data_quality(self, df: pd.DataFrame) -> ValidationResult:
        """Validate data quality."""
        errors = []
        warnings = []
        metrics = {}
        
        # Check for missing values
        missing_stats = self._check_missing_values(df)
        metrics.update(missing_stats['metrics'])
        
        if missing_stats['critical_missing']:
            errors.extend(missing_stats['errors'])
        warnings.extend(missing_stats['warnings'])
        
        # Check for duplicates
        duplicate_stats = self._check_duplicates(df)
        metrics.update(duplicate_stats['metrics'])
        warnings.extend(duplicate_stats['warnings'])
        
        # Check value ranges
        range_stats = self._check_value_ranges(df)
        errors.extend(range_stats['errors'])
        warnings.extend(range_stats['warnings'])
        metrics.update(range_stats['metrics'])
        
        # Check categorical values
        categorical_stats = self._check_categorical_values(df)
        warnings.extend(categorical_stats['warnings'])
        metrics.update(categorical_stats['metrics'])
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, metrics)
    
    def _check_missing_values(self, df: pd.DataFrame) -> Dict:
        """Check for missing values."""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        errors = []
        warnings = []
        
        required_columns = self.validation_rules.get('required_columns', [])
        for col in required_columns:
            if col in df.columns and missing_counts[col] > 0:
                errors.append(f"Required column {col} has {missing_counts[col]} missing values")
        
        # Warn about high missing value percentages
        for col, pct in missing_percentages.items():
            if pct > 30:  # More than 30% missing
                warnings.append(f"Column {col} has {pct:.2f}% missing values")
        
        return {
            'errors': errors,
            'warnings': warnings,
            'metrics': {
                'total_missing_values': missing_counts.sum(),
                'missing_percentage': (missing_counts.sum() / (len(df) * len(df.columns))) * 100
            },
            'critical_missing': len(errors) > 0
        }
    
    def _check_duplicates(self, df: pd.DataFrame) -> Dict:
        """Check for duplicate records."""
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(df)) * 100
        
        warnings = []
        if duplicate_count > 0:
            warnings.append(f"Found {duplicate_count} duplicate records ({duplicate_percentage:.2f}%)")
        
        return {
            'warnings': warnings,
            'metrics': {
                'duplicate_count': duplicate_count,
                'duplicate_percentage': duplicate_percentage
            }
        }
    
    def _check_value_ranges(self, df: pd.DataFrame) -> Dict:
        """Check if values are within expected ranges."""
        errors = []
        warnings = []
        metrics = {}
        
        # Check numeric ranges
        for col, rules in self.validation_rules.items():
            if isinstance(rules, dict) and 'min' in rules and 'max' in rules:
                if col in df.columns:
                    # Convert to numeric, handling non-numeric values
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    
                    # Check range violations
                    out_of_range = ((numeric_col < rules['min']) | 
                                   (numeric_col > rules['max'])) & numeric_col.notna()
                    
                    if out_of_range.any():
                        count = out_of_range.sum()
                        percentage = (count / len(df)) * 100
                        if percentage > 5:  # More than 5% out of range
                            errors.append(f"Column {col}: {count} values out of range "
                                         f"[{rules['min']}, {rules['max']}]")
                        else:
                            warnings.append(f"Column {col}: {count} values out of range "
                                          f"[{rules['min']}, {rules['max']}]")
                        
                        metrics[f'{col}_out_of_range_count'] = count
                        metrics[f'{col}_out_of_range_percentage'] = percentage
        
        return {
            'errors': errors,
            'warnings': warnings,
            'metrics': metrics
        }
    
    def _check_categorical_values(self, df: pd.DataFrame) -> Dict:
        """Check categorical values for consistency."""
        warnings = []
        metrics = {}
        
        categorical_columns = self.validation_rules.get('categorical_columns', [])
        
        for col in categorical_columns:
            if col in df.columns:
                unique_values = df[col].unique()
                unique_count = len(unique_values)
                
                # Check for unexpected high cardinality
                if unique_count > 50:  # Threshold for high cardinality
                    warnings.append(f"Column {col} has high cardinality: {unique_count} unique values")
                
                # Check for potential data quality issues
                if col in ['gender'] and unique_count > 3:
                    warnings.append(f"Column {col} has unexpected values: {unique_values}")
                
                metrics[f'{col}_unique_count'] = unique_count
        
        return {
            'warnings': warnings,
            'metrics': metrics
        }
    
    def validate_distribution(self, df: pd.DataFrame) -> ValidationResult:
        """Validate data distribution."""
        errors = []
        warnings = []
        metrics = {}
        
        # Check class distribution for target variable
        if 'Churn' in df.columns:
            churn_distribution = df['Churn'].value_counts(normalize=True)
            metrics['churn_rate'] = churn_distribution.get('Yes', 0) if 'Yes' in churn_distribution else churn_distribution.get(1, 0)
            
            # Check for class imbalance
            if metrics['churn_rate'] < 0.05 or metrics['churn_rate'] > 0.95:
                warnings.append(f"Severe class imbalance detected. Churn rate: {metrics['churn_rate']:.3f}")
            elif metrics['churn_rate'] < 0.1 or metrics['churn_rate'] > 0.9:
                warnings.append(f"Class imbalance detected. Churn rate: {metrics['churn_rate']:.3f}")
        
        # Check for outliers in numeric columns
        numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for col in numeric_columns:
            if col in df.columns:
                outlier_stats = self._detect_outliers(df[col])
                if outlier_stats['outlier_percentage'] > 10:  # More than 10% outliers
                    warnings.append(f"Column {col} has {outlier_stats['outlier_percentage']:.2f}% outliers")
                
                metrics[f'{col}_outlier_percentage'] = outlier_stats['outlier_percentage']
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, metrics)
    
    def _detect_outliers(self, series: pd.Series) -> Dict:
        """Detect outliers using IQR method."""
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        
        if len(numeric_series) == 0:
            return {'outlier_percentage': 0}
        
        Q1 = numeric_series.quantile(0.25)
        Q3 = numeric_series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((numeric_series < lower_bound) | (numeric_series > upper_bound))
        outlier_percentage = (outliers.sum() / len(numeric_series)) * 100
        
        return {'outlier_percentage': outlier_percentage}
    
    def validate_full_dataset(self, df: pd.DataFrame) -> ValidationResult:
        """Perform comprehensive data validation."""
        all_errors = []
        all_warnings = []
        all_metrics = {}
        
        # Schema validation
        schema_result = self.validate_schema(df)
        all_errors.extend(schema_result.errors)
        all_warnings.extend(schema_result.warnings)
        all_metrics.update(schema_result.metrics)
        
        if not schema_result.is_valid:
            logger.error("Schema validation failed. Skipping further validation.")
            return ValidationResult(False, all_errors, all_warnings, all_metrics)
        
        # Data quality validation
        quality_result = self.validate_data_quality(df)
        all_errors.extend(quality_result.errors)
        all_warnings.extend(quality_result.warnings)
        all_metrics.update(quality_result.metrics)
        
        # Distribution validation
        distribution_result = self.validate_distribution(df)
        all_errors.extend(distribution_result.errors)
        all_warnings.extend(distribution_result.warnings)
        all_metrics.update(distribution_result.metrics)
        
        # Overall validation result
        is_valid = len(all_errors) == 0
        
        # Add dataset-level metrics
        all_metrics.update({
            'dataset_size': len(df),
            'feature_count': len(df.columns),
            'validation_timestamp': datetime.now().isoformat()
        })
        
        return ValidationResult(is_valid, all_errors, all_warnings, all_metrics)
    
    def generate_validation_report(self, validation_result: ValidationResult, 
                                 output_path: str = "reports/data_quality/validation_report.txt"):
        """Generate a validation report."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("DATA VALIDATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Validation Status: {'PASSED' if validation_result.is_valid else 'FAILED'}\n")
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if validation_result.errors:
                f.write("ERRORS:\n")
                for error in validation_result.errors:
                    f.write(f"  - {error}\n")
                f.write("\n")
            
            if validation_result.warnings:
                f.write("WARNINGS:\n")
                for warning in validation_result.warnings:
                    f.write(f"  - {warning}\n")
                f.write("\n")
            
            f.write("METRICS:\n")
            for metric, value in validation_result.metrics.items():
                f.write(f"  {metric}: {value}\n")
        
        logger.info(f"Validation report saved to {output_path}")

def main():
    """Main function for testing data validation."""
    # Example usage
    validator = DataValidator()
    
    # Load sample data (you would load your actual dataset here)
    try:
        df = pd.read_csv("data/raw/Customer_Churn.csv")
        
        # Perform validation
        result = validator.validate_full_dataset(df)
        
        # Generate report
        validator.generate_validation_report(result)
        
        # Print summary
        print(f"Validation {'PASSED' if result.is_valid else 'FAILED'}")
        print(f"Errors: {len(result.errors)}")
        print(f"Warnings: {len(result.warnings)}")
        
    except FileNotFoundError:
        print("Sample data file not found. Please ensure data/raw/Customer_Churn.csv exists.")

if __name__ == "__main__":
    main()