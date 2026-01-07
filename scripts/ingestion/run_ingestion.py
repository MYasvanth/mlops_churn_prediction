import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_ingestion import DataIngestion
from omegaconf import OmegaConf

def main():
    """Run data ingestion pipeline"""
    try:
        # Load data configuration directly
        data_config_path = Path("configs/data/local.yaml")
        if not data_config_path.exists():
            raise FileNotFoundError(f"Data config file not found: {data_config_path}")
        
        data_config = OmegaConf.load(data_config_path)
        
        # Create data config object
        class SimpleDataConfig:
            def __init__(self, raw_data_path, processed_data_path, validation_rules=None):
                self.raw_data_path = raw_data_path
                self.processed_data_path = processed_data_path
                self.validation_rules = validation_rules or {}
        
        # Use configuration from file
        config = SimpleDataConfig(
            raw_data_path=data_config.data.raw_data_path,
            processed_data_path="data/processed/train.csv",  # Override to specific file
            validation_rules={}
        )
        
        # Run data ingestion
        ingestion = DataIngestion(config)
        df = ingestion.run_data_ingestion()
        
        print(f"✅ Data ingestion completed successfully!")
        print(f"   - Raw data shape: {df.shape}")
        print(f"   - Processed data saved to: {config.processed_data_path}")
        
    except Exception as e:
        print(f"❌ Data ingestion failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
