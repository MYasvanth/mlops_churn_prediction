# Data Schema

## Customer Table

| Field | Type | Description | PII |
|-------|------|-------------|-----|
| customer_id | VARCHAR(50) | Unique customer identifier | No |
| age | INT | Customer age | No |
| gender | VARCHAR(10) | Customer gender | No |
| location | VARCHAR(100) | Customer location | Yes |
| signup_date | DATE | Account creation date | No |
| subscription_type | VARCHAR(20) | Basic/Standard/Premium | No |
| contract_length | INT | Months in contract | No |

## Usage Metrics Table

| Field | Type | Description |
|-------|------|-------------|
| customer_id | VARCHAR(50) | Foreign key to customer |
| login_frequency | INT | Monthly login count |
| session_duration_avg | FLOAT | Average session time (minutes) |
| feature_count | INT | Number of features used |
| last_active_date | DATE | Most recent activity |
| inactive_days | INT | Days since last activity |

## Churn Label Table

| Field | Type | Description |
|-------|------|-------------|
| customer_id | VARCHAR(50) | Foreign key to customer |
| churn_date | DATE | Date of churn (null if active) |
| churn_label | BOOLEAN | 1 = Churned, 0 = Active |
| churn_reason | VARCHAR(100) | Reason for churn (if known) |

## Feature Engineering Output

| Feature Name | Type | Description |
|--------------|------|-------------|
| tenure_months | INT | Total months as customer |
| monthly_charges | FLOAT | Average monthly charges |
| total_charges | FLOAT | Lifetime total charges |
| payment_delay_count | INT | Number of late payments |
| support_ticket_count | INT | Number of support tickets |
| usage_score | FLOAT | Normalized usage intensity |
| risk_score | FLOAT | Calculated churn risk (0-1) |

## Data Types Summary

- **Numeric**: age, tenure_months, charges, usage metrics
- **Categorical**: gender, subscription_type, location
- **Temporal**: signup_date, last_active_date, churn_date
- **Boolean**: churn_label, payment_status
