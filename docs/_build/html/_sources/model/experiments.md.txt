# Experiment Tracking

## MLflow Integration

All experiments are tracked using MLflow:

- **Tracking URI**: Configured in `mlflow.yaml`
- **Experiment Name**: `churn_prediction`
- **Run Tags**: environment, model_type, data_version

## Experiment Results

### Experiment 1: Baseline - Logistic Regression

| Metric | Value |
|--------|-------|
| Accuracy | 0.78 |
| Precision | 0.72 |
| Recall | 0.68 |
| F1-Score | 0.70 |
| AUC-ROC | 0.81 |

### Experiment 2: Random Forest

| Metric | Value |
|--------|-------|
| Accuracy | 0.82 |
| Precision | 0.76 |
| Recall | 0.74 |
| F1-Score | 0.75 |
| AUC-ROC | 0.85 |

### Experiment 3: XGBoost

| Metric | Value |
|--------|-------|
| Accuracy | 0.84 |
| Precision | 0.79 |
| Recall | 0.77 |
| F1-Score | 0.78 |
| AUC-ROC | 0.87 |

### Experiment 4: LightGBM (Selected)

| Metric | Value |
|--------|-------|
| Accuracy | 0.85 |
| Precision | 0.81 |
| Recall | 0.79 |
| F1-Score | 0.80 |
| AUC-ROC | 0.88 |

## Hyperparameter Tuning

### LightGBM Best Parameters

```
python
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'class_weight': 'balanced'
}
```

## Feature Importance

Top 5 Features:
1. `inactive_days` - Days since last activity
2. `tenure_months` - Customer tenure
3. `support_ticket_count` - Number of support tickets
4. `monthly_charges` - Monthly subscription cost
5. `payment_delay_count` - Payment delays
