#!/usr/bin/env python3
"""
Script to analyze all staging models and their performance metrics
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.unified_model_registry_fixed import UnifiedModelRegistry

def analyze_staging_models():
    """Analyze all staging models and their performance"""
    try:
        registry = UnifiedModelRegistry()
        staging_models = registry.list_models("staging")
        
        print("📊 Staging Models Performance Analysis")
        print("=" * 60)
        print(f"Total staging models found: {len(staging_models)}")
        
        if not staging_models:
            print("No staging models found!")
            return
        
        models_by_type = {}
        
        for model in staging_models:
            model_type = model.get('model_type', 'unknown')
            metrics = model.get('performance_metrics', {})
            
            if model_type not in models_by_type:
                models_by_type[model_type] = []
            
            models_by_type[model_type].append({
                'model_id': model['model_id'],
                'f1_score': metrics.get('f1_score', 0),
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'roc_auc': metrics.get('roc_auc', 0)
            })
        
        # Print analysis by model type
        for model_type, models in models_by_type.items():
            print(f"\n🔍 {model_type.upper()} Models ({len(models)}):")
            print("-" * 40)
            
            # Sort by F1 score descending
            models.sort(key=lambda x: x['f1_score'], reverse=True)
            
            for i, model in enumerate(models, 1):
                print(f"{i:2d}. {model['model_id']}")
                print(f"    F1: {model['f1_score']:.4f} | Acc: {model['accuracy']:.4f} | "
                      f"Prec: {model['precision']:.4f} | Rec: {model['recall']:.4f} | "
                      f"AUC: {model['roc_auc']:.4f}")
        
        # Find overall best model
        all_models = [model for models in models_by_type.values() for model in models]
        best_model = max(all_models, key=lambda x: x['f1_score'])
        
        print(f"\n🏆 Best Overall Model: {best_model['model_id']}")
        print(f"   F1 Score: {best_model['f1_score']:.4f}")
        print(f"   Model Type: {next((k for k, v in models_by_type.items() if best_model in v), 'unknown')}")
        
    except Exception as e:
        print(f"Error analyzing staging models: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_staging_models()
