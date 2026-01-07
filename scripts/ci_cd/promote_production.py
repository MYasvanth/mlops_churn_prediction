#!/usr/bin/env python3
"""Promote model to production"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append('src')

from src.models.unified_model_registry_fixed import UnifiedModelRegistry

def promote_to_production(model_id):
    """Promote model from staging to production"""
    try:
        registry = UnifiedModelRegistry()
        
        # Promote model
        success = registry.promote_model(model_id, 'staging', 'production')
        
        if success:
            print(f"Model {model_id} promoted to production")
            return True
        else:
            print(f"Failed to promote model {model_id}")
            return False
            
    except Exception as e:
        print(f"Promotion failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python promote_production.py <model_id>")
        sys.exit(1)
        
    model_id = sys.argv[1]
    success = promote_to_production(model_id)
    sys.exit(0 if success else 1)