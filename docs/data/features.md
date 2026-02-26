# Feature Engineering

## Feature Categories

### 1. Customer Demographics
- Age
- Gender
- Location (encoded)
- Contract type

### 2. Account Features
- Tenure (months since signup)
- Subscription type
- Monthly charges
- Total charges
- Payment history

### 3. Behavioral Features
- Login frequency
- Session duration
- Feature usage count
- Inactivity period
- Activity trend

### 4. Engagement Features
- Support ticket count
- Response time
- Resolution satisfaction
- Communication frequency

## Feature Engineering Pipeline

```
python
# Example feature engineering code
def create_features(df):
    # Tenure features
    df['tenure_months'] = (df['current_date'] - df['signup_date']).dt.days / 30
    
    # Usage features
    df['usage_score'] = df['login_frequency'] * df['session_duration_avg']
    
    # Risk indicators
    df['payment_delay_flag'] = (df['payment_delay_count'] > 2).astype(int)
    df['inactivity_flag'] = (df['inactive_days'] > 30).astype(int)
    
    return df
```

## Feature Transformations

| Feature | Transformation | Reason |
|---------|---------------|--------|
| age | Binning | Non-linear relationship with churn |
| location | Target encoding | High cardinality |
| monthly_charges | Log transform | Skewed distribution |
| session_duration | Standardization | Different scales |

## Feature Selection

### Importance-Based Selection
Using XGBoost feature importance:
1. Train initial model
2. Rank features by importance
3. Select top 20 features
4. Retrain with selected features

### Correlation Analysis
- Remove highly correlated features (|r| > 0.9)
- Keep features with correlation to target > 0.1

## Feature Store

Features are stored in the feature store for:
- Consistent training/serving
- Feature sharing across models
- Point-in-time correctness
