# Data Sources

## Primary Data Sources

### 1. Customer Data
- **Source**: Customer database (SQL)
- **Description**: Customer demographic and account information
- **Update Frequency**: Daily
- **Schema**: See [Data Schema](./schema.md)

### 2. Transaction Data
- **Source**: Payment/Billing system
- **Description**: Customer payment history and transaction details
- **Update Frequency**: Real-time
- **Key Fields**: transaction_id, customer_id, amount, date, status

### 3. Usage Data
- **Source**: Product analytics platform
- **Description**: Customer product usage patterns
- **Update Frequency**: Hourly
- **Key Metrics**: login frequency, feature usage, session duration

### 4. Support Data
- **Source**: Customer support system
- **Description**: Support tickets and customer interactions
- **Update Frequency**: Real-time
- **Key Fields**: ticket_id, customer_id, type, status, resolution_time

## Data Ingestion Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Source    │───▶│  Ingestion  │───▶│   Raw Data  │
│  Systems    │    │   Pipeline   │    │   Storage   │
└─────────────┘    └─────────────┘    └─────────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │  Data Validation │
                                    │     & Cleaning   │
                                    └─────────────────┘
```

## Data Storage

- **Raw Data**: S3/GCS bucket (`s3://mlops-churn-data/raw/`)
- **Processed Data**: Snowflake/BigQuery (`churn_data.processed`)
- **Feature Store**: Redis/Feast (`features.churn_features`)

## Access Requirements

| Data Source | Team Access | Security Level |
|-------------|-------------|-----------------|
| Customer Data | All Teams | PII - Encrypted |
| Transaction Data | Finance, DS | PCI - Highly Secure |
| Usage Data | Product, DS | Internal |
| Support Data | Support, DS | Internal |
