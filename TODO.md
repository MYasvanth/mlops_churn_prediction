# Config Consistency Implementation Plan

## Tasks to Complete

- [x] Update params.yaml: Change raw_data_path from "WA_Fn-UseC-Telco-Customer-Churn.csv" to "data/raw/Customer_data.csv"
- [x] Update dvc.yaml: Change deps path from "data/raw/customer_data.csv" to "data/raw/Customer_data.csv"
- [x] Fix configs/deployment/deployment_config.yaml: Remove duplicate environment variables
- [x] Update src/utils/config_loader.py: Standardize config loading to use Hydra consistently
- [x] Clean up empty config directories: Remove or populate configs/training/ and configs/experiment/
- [x] Test config loading across components
- [x] Run training pipeline to verify no path errors
- [x] Validate deployment configs work correctly
