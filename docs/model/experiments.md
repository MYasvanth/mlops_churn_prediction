# Experiment Tracking

## MLflow Integration

All experiments are tracked using MLflow:

- **Tracking URI**: `file:./mlruns`
- **Experiment Name**: `churn_prediction`
- **Run Tags**: environment, model_type, data_version

## Experiment Results

**Note**: This project demonstrates MLOps infrastructure with hyperparameter tuning experiments. The following shows actual results from the implemented system.

### Hyperparameter Optimization Experiments

**Method**: Bayesian optimization using Optuna
**Optimization Metric**: Cross-validation score
**Cross-Validation**: 5-fold stratified

Multiple hyperparameter tuning runs were conducted:

| Run ID | Model | Best CV Score | Status |
|--------|-------|---------------|--------|
| auspicious-moth-513 | XGBoost | 0.7992 | ✅ Completed |
| treasured-moose-414 | XGBoost | 0.7992 | ✅ Completed |
| shivering-ram-427 | XGBoost | 0.7992 | ✅ Completed |
| grandiose-lark-915 | XGBoost | 0.7984 | ✅ Completed |
| blushing-seal-234 | XGBoost | 0.7947 | ✅ Completed |

### Random Forest - Full Evaluation

**Test Set Performance** (200 samples):

| Metric | Value |
|--------|-------|
| Accuracy | 0.925 |
| Precision | 0.925 |
| Recall | 0.925 |
| F1-Score | 0.925 |
| AUC-ROC | 0.972 |

**Confusion Matrix**:
```
                Predicted
                No    Yes
Actual  No      93     7
        Yes      8    92
```

**Bias-Variance Analysis**:
- Train Score: 1.00
- Validation Score: 0.84
- Train-Val Gap: 0.16 (16%)
- Diagnosis: HIGH_VARIANCE_OVERFIT
- Recommendation: Reduce model complexity, add regularization, get more data

## XGBoost Hyperparameter Tuning

**Best Parameters** (from Optuna optimization):

```python
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0,
    'random_state': 42
}
```

## Infrastructure Capabilities

### Implemented Features
- ✅ MLflow experiment tracking (19 runs logged)
- ✅ Optuna hyperparameter optimization
- ✅ Cross-validation framework
- ✅ Model evaluation with bias-variance analysis
- ✅ Confusion matrix and classification reports
- ✅ Learning curve generation

### Model Registry
- Multiple XGBoost models in staging
- Random Forest with full evaluation metrics
- Production deployment pipeline ready

## Key Findings

1. **XGBoost Optimization**: Achieved ~79.9% CV score through hyperparameter tuning
2. **Random Forest**: High test accuracy (92.5%) but shows overfitting (16% train-val gap)
3. **MLOps Infrastructure**: Fully functional experiment tracking and model registry
4. **Production Ready**: API, monitoring, and deployment infrastructure in place
