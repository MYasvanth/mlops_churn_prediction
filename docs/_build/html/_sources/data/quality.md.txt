# Data Quality

## Quality Framework

### Data Quality Dimensions

| Dimension | Description | Target |
|-----------|-------------|--------|
| **Completeness** | No missing values | > 99% |
| **Accuracy** | Correct values | > 98% |
| **Consistency** | Uniform format | 100% |
| **Timeliness** | Up-to-date data | < 24h latency |
| **Uniqueness** | No duplicates | 100% |

## Validation Rules

### Schema Validation
- Data types match expected schema
- Required fields are present
- Value ranges are within bounds

### Business Rules
- Customer age: 18-100
- Tenure: >= 0
- Charges: >= 0
- Dates: Not in future

## Monitoring

### Data Quality Checks

```
python
# Example data quality check
from great_expectations import GreatExpectations

ge = GreatExpectations()
expectation_suite = ge.get_expectation_suite("churn_data")

# Run validation
results = ge.validate(
    batch_request=batch_request,
    expectation_suite=expectation_suite
)
```

### Alerting

| Check | Alert Threshold |
|-------|----------------|
| Missing values | > 1% |
| Duplicate records | > 0.1% |
| Invalid values | > 0.5% |
| Stale data | > 48 hours |

## Data Quality Dashboard

- Great Expectations dashboard
- Data lineage tracking
- Anomaly detection
- Historical trends

## Remediation Process

1. **Detection**: Automated alerts trigger
2. **Investigation**: Root cause analysis
3. **Correction**: Fix data at source
4. **Validation**: Re-run quality checks
5. **Documentation**: Log incident and resolution
