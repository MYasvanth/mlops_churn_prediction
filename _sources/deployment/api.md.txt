# API Documentation

## Base URL

```
Production: https://api.churn-prediction.example.com/v1
Staging: https://staging-api.churn-prediction.example.com/v1
```

## Authentication

All requests require a Bearer token:

```
Authorization: Bearer <your-api-key>
```

## Endpoints

### 1. Predict Churn

**POST** `/predict`

Request:
```
json
{
  "customer_id": "CUST12345",
  "features": {
    "age": 35,
    "tenure_months": 24,
    "monthly_charges": 79.99,
    "total_charges": 1919.76,
    "login_frequency": 15,
    "session_duration_avg": 25.5,
    "support_ticket_count": 2,
    "payment_delay_count": 0,
    "inactive_days": 5
  }
}
```

Response:
```
json
{
  "customer_id": "CUST12345",
  "churn_probability": 0.73,
  "risk_level": "HIGH",
  "prediction_timestamp": "2024-01-15T10:30:00Z",
  "model_version": "lightgbm_v2.3.1"
}
```

### 2. Batch Prediction

**POST** `/predict/batch`

Request:
```
json
{
  "customers": [
    {"customer_id": "CUST12345", "features": {...}},
    {"customer_id": "CUST12346", "features": {...}}
  ]
}
```

Response:
```
json
{
  "predictions": [
    {"customer_id": "CUST12345", "churn_probability": 0.73, "risk_level": "HIGH"},
    {"customer_id": "CUST12346", "churn_probability": 0.12, "risk_level": "LOW"}
  ],
  "processing_time_ms": 250
}
```

### 3. Get Model Info

**GET** `/model/info`

Response:
```
json
{
  "model_name": "lightgbm_churn",
  "model_version": "2.3.1",
  "training_date": "2024-01-10",
  "metrics": {
    "auc_roc": 0.88,
    "f1_score": 0.80,
    "precision": 0.81,
    "recall": 0.79
  },
  "features": ["age", "tenure_months", ...]
}
```

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Invalid request format |
| 401 | Unauthorized - Invalid API key |
| 429 | Rate limit exceeded |
| 500 | Internal server error |
| 503 | Service unavailable |

## Rate Limits

- **Standard**: 100 requests/minute
- **Premium**: 1000 requests/minute
- **Batch**: 10 requests/minute (max 1000 customers per request)

## SDKs

### Python
```
python
from churn_client import ChurnPredictor

client = ChurnPredictor(api_key="your-api-key")
result = client.predict(customer_id="CUST12345", features={...})
```

### JavaScript
```javascript
const churnClient = require('churn-sdk');

const result = await churnClient.predict({
  customerId: 'CUST12345',
  features: {...}
});
