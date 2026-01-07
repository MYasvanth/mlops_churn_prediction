#!/usr/bin/env python3
"""
Script to promote the best performing model from staging to production
"""

import sys
from pathlib import Path
import json
import joblib
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.unified_model_registry_fixed import UnifiedModelRegistry

def find_best_staging_model():
    """Find the best performing model in staging based on F1 score"""
    registry = UnifiedModelRegistry()
    staging_models = registry.list_models("staging")
    
    if not staging_models:
        print("No staging models found!")
        return None
    
    # Find model with highest F1 score
    best_model = None
    best_f1 = -1
    
    for model in staging_models:
        metrics = model.get('performance_metrics', {})
        f1_score = metrics.get('f1_score', 0)
        
        if f1_score > best_f1:
            best_f1 = f1_score
            best_model = model
    
    return best_model

def promote_model_to_production(model_info):
    """Promote a model to production stage"""
    registry = UnifiedModelRegistry()
    
    model_id = model_info['model_id']
    print(f"Promoting model {model_id} to production...")
    
    # Load the model from staging
    model = registry.load_model(model_id, "staging")
    
    # Register it in production
    metadata = {
        'model_name': model_info.get('model_name', 'churn_model'),
        'model_type': model_info.get('model_type', 'unknown'),
        'performance_metrics': model_info.get('performance_metrics', {}),
        'hyperparameters': model_info.get('hyperparameters', {}),
        'feature_columns': model_info.get('feature_columns', []),
        'created_by': model_info.get('created_by', 'promotion_script'),
        'description': f"Promoted to production from staging on {datetime.now().isoformat()}"
    }
    
    new_model_id = registry.register_model(model, metadata, "production")
    print(f"Model successfully promoted to production with ID: {new_model_id}")
    return new_model_id

def main():
    """Main function to promote best model to production"""
    print("Finding best staging model...")
    
    best_model = find_best_staging_model()
    if not best_model:
        print("No suitable model found for promotion")
        return
    
    print(f"Best model found: {best_model['model_id']}")
    print(f"Performance metrics: {best_model['performance_metrics']}")
    
    # Promote to production
    new_model_id = promote_model_to_production(best_model)
    
    # Verify promotion
    registry = UnifiedModelRegistry()
    production_models = registry.list_models("production")
    print(f"\nProduction models after promotion: {len(production_models)}")
    
    for model in production_models:
        print(f"  - {model['model_id']}: {model['performance_metrics'].get('f1_score', 0):.4f} F1")

if __name__ == "__main__":
    main()
