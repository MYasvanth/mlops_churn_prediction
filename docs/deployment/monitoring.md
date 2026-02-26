# Monitoring Setup

## Monitoring Stack

| Component | Purpose |
|-----------|---------|
| **Prometheus** | Metrics collection |
| **Grafana** | Visualization & dashboards |
| **Alertmanager** | Alert routing |
| **ELK Stack** | Log aggregation |

## Key Metrics

### Model Performance Metrics
- `prediction_latency_p50` - 50th percentile latency
- `prediction_latency_p95` - 95th percentile latency
- `prediction_latency_p99` - 99th percentile latency
- `prediction_count` - Total predictions made
- `prediction_errors` - Failed predictions

### Business Metrics
- `churn_rate_actual` - Actual churn rate
- `retention_campaign_impact` - Campaign effectiveness
- `revenue_at_risk` - Predicted revenue at risk

### Data Quality Metrics
- `data_freshness` - Time since last data update
- `missing_values_count` - Missing feature values
- `drift_score` - Population Stability Index (PSI)

## Dashboards

### 1. Executive Dashboard
- Churn trend over time
- Model performance KPIs
- Business impact metrics
- ROI tracking

### 2. Operations Dashboard
- API health & uptime
- Latency percentiles
- Error rates
- Resource utilization

### 3. Model Performance
- AUC-ROC over time
- Precision/Recall trends
- Feature importance drift
- Prediction distribution

## Alerting Rules

| Alert | Condition | Severity |
|-------|-----------|----------|
| High Latency | p95 > 200ms | Warning |
| Critical Latency | p99 > 500ms | Critical |
| Model Drift | PSI > 0.20 | Warning |
| High Error Rate | Errors > 1% | Critical |
| Data Stale | Freshness > 24h | Warning |

## Logging

### Structured Log Format
```
json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "service": "prediction-api",
  "customer_id": "CUST12345",
  "prediction": 0.73,
  "latency_ms": 45,
  "model_version": "lightgbm_v2.3.1"
}
```

### Log Retention
- **Development**: 7 days
- **Staging**: 30 days
- **Production**: 1 year

## Retraining Triggers

Automated retraining is triggered when:
1. Model drift PSI > 0.10
2. Performance drops > 5%
3. Scheduled (weekly/monthly)
4. Manual trigger

## On-Call

- **Primary**: DevOps Team
- **Secondary**: Data Science Team
- **Escalation**: Engineering Manager
