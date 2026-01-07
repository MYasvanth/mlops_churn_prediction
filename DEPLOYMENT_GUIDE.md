# MLOps Churn Prediction - Complete Deployment Guide

This guide covers the complete deployment and monitoring setup for the MLOps churn prediction system.

## 🚀 Quick Start Deployment

### 1. Prerequisites
```bash
# Install required dependencies
pip install -r requirements.txt
pip install omegaconf evidently streamlit optuna-dashboard

# Verify installation
python scripts/simple_monitoring_test.py
```

### 2. Complete Deployment Setup
```bash
# Run the complete deployment script
python scripts/start_complete_deployment.py

# Alternative: Start services individually
python scripts/run_fastapi_server.py &          # API Server (port 8000)
python -m streamlit run src/deployment/streamlit_app.py --server.port 8501 &  # Dashboard
mlflow ui --port 5000 &                         # MLflow Tracking
optuna-dashboard --port 8080 sqlite:///optuna_studies.db &  # Optuna Dashboard
```

### 3. Test the Deployment
```bash
# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/models

# Test monitoring
python scripts/run_monitoring.py
```

## 📊 Service Endpoints

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| FastAPI | 8000 | http://localhost:8000 | REST API for predictions |
| Streamlit | 8501 | http://localhost:8501 | Interactive dashboard |
| MLflow | 5000 | http://localhost:5000 | Experiment tracking |
| Optuna | 8080 | http://localhost:8080 | Hyperparameter optimization |

## 🔧 Configuration Files

### Monitoring Configuration (`configs/monitoring/monitoring_config.yaml`)
```yaml
data_drift:
  drift_threshold: 0.5
  confidence_level: 0.95
  stattest_threshold: 0.1
  target_column: churn
  prediction_column: prediction
```

### Deployment Configuration (`configs/deployment/deployment_config.yaml`)
```yaml
deployment:
  stages:
    staging:
      replicas: 2
      resources:
        cpu: "500m"
        memory: "1Gi"
    production:
      replicas: 3
      resources:
        cpu: "2"
        memory: "4Gi"
```

## 🐳 Docker Deployment

### Build Docker Image
```bash
cd deployment/docker
docker build -t churn-prediction:latest .
```

### Run with Docker Compose
```bash
docker-compose up -d
```

### Kubernetes Deployment
```bash
# Apply Kubernetes manifests
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml
```

## 📈 Monitoring Setup

### 1. Initialize Monitoring
```bash
python scripts/setup_monitoring.py
```

### 2. Run Monitoring
```bash
# Single monitoring run
python scripts/run_monitoring.py

# Continuous monitoring (every 30 minutes)
python scripts/run_monitoring.py --mode continuous --interval 30
```

### 3. Monitor Alerts
Check the alert system logs and database for monitoring events:
```bash
# View alert history
sqlite3 monitoring/alerts.db "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 10;"
```

## 🧪 Testing

### API Testing
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test prediction endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1,2,3,4,5,6,7,8,9,10], "model_id": "xgboost_20250820_104603"}'
```

### Model Testing
```bash
# Test model loading and prediction
python scripts/test_model_endpoints.py

# Quick model test
python scripts/quick_model_test.py
```

## 🔄 CI/CD Pipeline

The project includes GitHub Actions configuration for automated testing and deployment:

```yaml
# .github/workflows/ci-cd.yaml
name: CI/CD Pipeline
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python -m pytest tests/ -v
```

## 📊 Performance Monitoring

### Key Metrics Tracked
- **Data Drift**: Statistical tests for feature distribution changes
- **Model Performance**: Accuracy, Precision, Recall, F1, ROC AUC
- **System Metrics**: Response time, throughput, error rates
- **Business Metrics**: Churn rate predictions vs actuals

### Alert Thresholds
- **Critical**: Performance degradation > 10%
- **High**: Data drift score > 0.3
- **Medium**: Single metric below threshold
- **Low**: Warning-level anomalies

## 🛠️ Troubleshooting

### Common Issues

1. **Dependency Issues**
   ```bash
   # Reinstall dependencies
   pip install --upgrade -r requirements.txt
   pip install omegaconf evidently
   ```

2. **Port Conflicts**
   ```bash
   # Check running services
   netstat -tulpn | grep :8000
   
   # Kill process on port
   lsof -ti:8000 | xargs kill -9
   ```

3. **Database Issues**
   ```bash
   # Recreate monitoring database
   rm monitoring/alerts.db
   python scripts/setup_monitoring.py
   ```

4. **Model Loading Issues**
   ```bash
   # Check model files
   ls -la models/production/
   
   # Test model loading
   python scripts/check_model_working.py
   ```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python scripts/run_monitoring.py --verbose
```

## 📋 Health Checks

### System Health
```bash
# Check all services
curl http://localhost:8000/health
curl http://localhost:8501/_stcore/health
curl http://localhost:5000/health
```

### Database Health
```bash
# Check monitoring database
sqlite3 monitoring/alerts.db ".tables"
sqlite3 monitoring/alerts.db "SELECT COUNT(*) FROM alerts;"
```

### Model Health
```bash
# Test model predictions
python scripts/test_prediction_endpoint.py

# Validate model alignment
python scripts/validate_model_alignment.py
```

## 🔮 Advanced Deployment

### Cloud Deployment (AWS)
```bash
# Deploy to EKS
eksctl create cluster --name churn-cluster
kubectl apply -f deployment/kubernetes/

# Deploy to ECS
aws ecs create-service --cluster churn-cluster --task-definition churn-task
```

### Monitoring Stack
```bash
# Deploy Prometheus
helm install prometheus prometheus-community/prometheus

# Deploy Grafana
helm install grafana grafana/grafana

# Configure datasources
kubectl apply -f monitoring/prometheus/
kubectl apply -f monitoring/grafana/
```

## 📞 Support

For deployment issues:
1. Check the troubleshooting section
2. Review service logs in `logs/` directory
3. Verify configuration files in `configs/`
4. Test individual components with provided scripts

The system is designed for production readiness with comprehensive monitoring, alerting, and deployment capabilities.
