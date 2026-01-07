# MLOps Churn Prediction - Monitoring System

This document describes the comprehensive monitoring system implemented for the MLOps churn prediction project.

## 📊 Monitoring Components

### 1. Data Drift Monitoring
- **Location**: `src/monitoring/data_drift_monitor.py`
- **Purpose**: Detects changes in data distribution between training and production data
- **Metrics**: KS Test, PSI, Wasserstein Distance
- **Alerts**: Automatic alerts when drift exceeds configured thresholds

### 2. Model Performance Monitoring
- **Location**: `src/monitoring/model_performance_monitor.py`
- **Purpose**: Tracks model performance metrics and detects degradation
- **Metrics**: Accuracy, Precision, Recall, F1 Score, ROC AUC
- **Alerts**: Performance degradation alerts with severity levels

### 3. Alert System
- **Location**: `src/monitoring/alert_system.py`
- **Purpose**: Manages and sends alerts through multiple channels
- **Channels**: Email, Slack integration
- **Features**: Rate limiting, severity levels, alert management

### 4. Optuna Monitoring
- **Location**: `src/monitoring/optuna_monitor.py`
- **Purpose**: Tracks hyperparameter optimization studies
- **Features**: Study management, visualization, optimization history

## 🚀 Quick Start

### 1. Setup Monitoring Infrastructure
```bash
# Run monitoring setup
python scripts/setup_monitoring.py

# Or run setup without database creation
python scripts/setup_monitoring.py --skip-db
```

### 2. Run Monitoring
```bash
# Single monitoring run
python scripts/run_monitoring.py

# Continuous monitoring (every 30 minutes)
python scripts/run_monitoring.py --mode continuous --interval 30

# With custom config
python scripts/run_monitoring.py --config configs/monitoring/custom_config.yaml
```

### 3. Start Complete System
```bash
# Start all services (MLflow, FastAPI, Streamlit, Optuna, Monitoring)
python scripts/start_complete_deployment.py

# Start specific services only
python scripts/start_complete_deployment.py --services mlflow fastapi streamlit

# Start without monitoring
python scripts/start_complete_deployment.py --no-monitoring
```

## 📋 Configuration

### Monitoring Configuration File
Located at `configs/monitoring/monitoring_config.yaml`:

```yaml
# Data Drift Monitoring
data_drift_monitoring:
  enabled: true
  drift_threshold: 0.15
  check_interval_minutes: 60

# Performance Monitoring  
performance_monitoring:
  enabled: true
  thresholds:
    min_accuracy: 0.8
    min_precision: 0.75
    degradation_threshold: 0.05

# Alert System
alerts:
  email:
    enabled: true
    smtp_server: "smtp.gmail.com"
    recipients: ["admin@company.com"]
  slack:
    enabled: false
    webhook_url: "your-slack-webhook"
```

## 🎯 Streamlit Dashboard

The Streamlit dashboard provides a comprehensive interface for:

### Model Prediction
- Interactive customer data input form
- Real-time churn predictions
- Feature importance visualization
- Confidence scoring

### Performance Monitoring
- Real-time performance metrics
- Data drift status
- Alert summaries
- Performance trend charts

### Optuna Dashboard
- Hyperparameter optimization studies
- Best parameters visualization
- Optimization progress tracking

### Data Analysis
- Dataset overview and statistics
- Feature distributions
- Correlation analysis
- Target distribution visualization

## 🔧 Customization

### Adding New Alert Types
1. Extend `AlertType` enum in `alert_system.py`
2. Add corresponding alert handling in monitoring components
3. Update configuration with new alert rules

### Custom Monitoring Metrics
1. Modify `PerformanceMetrics` dataclass
2. Extend `calculate_metrics` method
3. Update threshold configuration

### Additional Notification Channels
1. Implement new notifier class inheriting from base
2. Add configuration support
3. Integrate with AlertManager

## 📈 Monitoring Reports

### Performance Reports
- Location: `reports/performance_reports/`
- Format: JSON with comprehensive metrics
- Retention: 30 days (configurable)

### Drift Reports  
- Location: `reports/drift_reports/`
- Format: JSON with drift analysis
- Retention: 30 days (configurable)

### Alert History
- Database: `monitoring/alerts.db`
- Tables: alerts, performance_metrics, drift_reports
- Queryable history of all monitoring events

## 🚨 Alert Examples

### Data Drift Alert
```
🚨 DATA_DRIFT - HIGH Severity
Drift detected with score 0.28
Drifted columns: ['monthly_charges', 'tenure']
```

### Performance Degradation Alert  
```
⚠️ MODEL_DEGRADATION - MEDIUM Severity  
Accuracy degraded by 7.2% (0.85 → 0.79)
```

### Threshold Breach Alert
```
🔴 THRESHOLD_BREACH - CRITICAL Severity  
Recall (0.65) below minimum threshold (0.70)
```

## 🛠️ Troubleshooting

### Common Issues

1. **Email Notifications Not Working**
   - Check SMTP configuration
   - Verify app passwords for email services
   - Test with `alert_system.py` test method

2. **Database Connection Issues**
   - Verify SQLite database path
   - Check file permissions

3. **MLflow Connection Issues**
   - Ensure MLflow server is running
   - Check tracking URI configuration

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python scripts/run_monitoring.py --verbose
```

## 📊 Monitoring Dashboard URLs

- **Streamlit Dashboard**: http://localhost:8501
- **MLflow Tracking**: http://localhost:5000  
- **FastAPI Docs**: http://localhost:8000/docs
- **Optuna Dashboard**: http://localhost:8080

## 🔮 Future Enhancements

- [x] Automated model retraining triggers (Implemented via .github/workflows/retrain.yml)
- [ ] Real-time streaming data monitoring
- [ ] Advanced anomaly detection
- [ ] Integration with Prometheus/Grafana
- [ ] Custom alert webhook support
- [ ] Multi-model comparison monitoring
- [ ] Business metric integration
- [ ] Automated report generation
