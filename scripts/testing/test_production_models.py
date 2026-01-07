#!/usr/bin/env python3
"""
Test script to verify production models are available
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.unified_model_registry_fixed import UnifiedModelRegistry

def main():
    """Test if production models are available"""
    try:
        registry = UnifiedModelRegistry()
        production_models = registry.list_models('production')
        
        print(f"Production models found: {len(production_models)}")
        
        for model in production_models:
            print(f"  - {model['model_id']}")
            print(f"    F1 Score: {model['performance_metrics']['f1_score']:.4f}")
            
        return len(production_models) > 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
