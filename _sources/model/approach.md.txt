# Algorithm Selection

## Candidate Algorithms

### 1. Logistic Regression
- **Pros**: Interpretable, fast, baseline model
- **Cons**: Limited predictive power for complex patterns
- **Use Case**: Baseline model, regulatory requirements

### 2. Random Forest
- **Pros**: Handles non-linear relationships, feature importance
- **Cons**: Can overfit, less interpretable
- **Use Case**: Quick iteration, feature selection

### 3. XGBoost / LightGBM (Gradient Boosting)
- **Pros**: High accuracy, handles imbalanced data, built-in regularization
- **Cons**: Requires tuning, less interpretable
- **Use Case**: Production model, best performance

### 4. Neural Networks
- **Pros**: Can capture complex patterns
- **Cons**: Requires more data, harder to train
- **Use Case**: If other models underperform

## Selected Approach

### Primary Model: LightGBM
- Best balance of performance and speed
- Native handling of categorical features
- Built-in support for imbalanced data

### Backup Model: XGBoost
- Alternative for ensemble
- Different regularization approach

## Model Architecture

```
┌─────────────────────────────────────────────┐
│              Input Features                  │
│  (Customer demographics, behavior, usage)   │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│           Feature Preprocessing              │
│  (Encoding, scaling, feature engineering)   │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│              LightGBM Model                  │
│  - 500 trees                                │
│  - Max depth: 6                             │
│  - Learning rate: 0.05                     │
│  - Class weight: balanced                   │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│            Output: Churn Probability         │
│              (0.0 - 1.0)                    │
└─────────────────────────────────────────────┘
```

## Hyperparameter Tuning

- Bayesian optimization with Optuna
- Cross-validation: 5-fold stratified
- Optimization metric: F1-score
