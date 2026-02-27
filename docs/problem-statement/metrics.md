# Success Metrics

## Primary Metrics

### Churn Prediction Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **Precision** | > 0.80 | Of all predicted churners, what % actually churned |
| **Recall** | > 0.75 | Of all actual churners, what % did we identify |
| **F1-Score** | > 0.77 | Harmonic mean of precision and recall |
| **AUC-ROC** | > 0.85 | Ability to distinguish between churners and non-churners |

### Business Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **Churn Rate Reduction** | 15-20% | Percentage reduction in overall churn |
| **Retention Cost Savings** | $X per retained customer | Cost saved by retaining vs acquiring new customers |
| **Prediction Latency** | < 100ms | Time to get a single prediction |

## Secondary Metrics

- **Data Quality Score**: > 0.95
- **Model Drift Threshold**: < 0.10 PSI
- **Feature Importance Stability**: Top 5 features remain consistent
- **API Availability**: > 99.5%

## Monitoring Alerts

- Model AUC drops below 0.80
- Data drift PSI exceeds 0.20
- Prediction latency exceeds 200ms
- API error rate exceeds 1%
