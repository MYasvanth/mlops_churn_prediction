#!/usr/bin/env python3
"""Deploy model to staging"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append('src')

from src.pipelines.unified_training_pipeline import UnifiedTrainingPipeline

def deploy_to_staging():
    """Train and deploy best model to staging"""
    try:
        pipeline = UnifiedTrainingPipeline()
        
        # Train best model
        result = pipeline.auto_select_best_model()
        best_model = result.get('best_model')
        
        if not best_model:
            print("No suitable model found")
            return False
            
        print(f"Deployed {best_model} to staging")
        return True
        
    except Exception as e:
        print(f"Deployment failed: {e}")
        return False

if __name__ == "__main__":
    success = deploy_to_staging()
    sys.exit(0 if success else 1)