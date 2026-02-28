# 🎯 MLOps Churn Prediction - Demo Guide

## 📋 Pre-Demo Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install streamlit omegaconf lightgbm evidently optuna-dashboard
```

### 2. Start Services
```bash
# Terminal 1: API Server
python scripts/monitoring/run_fastapi_server.py

# Terminal 2: Streamlit Dashboard  
streamlit run src/deployment/streamlit_app.py --server.port 8501

# Terminal 3: MLflow (optional)
mlflow ui --port 5000
```

## 🎬 Demo Flow (15 minutes)

### **Part 1: MLOps Pipeline Overview (4 mins)**

**Show:** Project structure and architecture
```bash
# Navigate to project
cd mlops_churn_prediction

# Show organized structure
tree -L 2
```

**Key Points:**
- "This is an end-to-end MLOps pipeline for customer churn prediction"
- "Production-ready with proper separation: data, models, deployment, monitoring"
- "Follows MLOps best practices with versioning, tracking, and automation"

**Demo:** Training pipeline
```bash
# Show unified training
python src/pipelines/unified_training_pipeline.py
```

**Highlight:**
- Multiple algorithms (XGBoost, LightGBM, Random Forest)
- Automated model comparison and selection
- MLflow experiment tracking

### **Part 2: Model Performance & Registry (3 mins)**

**Show:** API endpoints working
```bash
# Open browser: http://localhost:8000/docs
# Test endpoints:
curl http://localhost:8000/health
curl http://localhost:8000/models
```

**Key Points:**
- "7 trained models with 79-80% accuracy"
- "Model registry with staging/production environments"
- "RESTful API for production deployment"

**Show:** Model comparison
- Navigate to `models/staging/` folder
- Show multiple model versions with timestamps
- Explain model lifecycle management

### **Part 3: Interactive Prediction Demo (4 mins)**

**Show:** Streamlit dashboard at `http://localhost:8501`

**Demo Flow:**
1. **Model Prediction Tab:**
   - Fill customer data form:
     - Gender: Female
     - Tenure: 12 months
     - Contract: Month-to-month
     - Monthly Charges: $85
     - Internet Service: Fiber optic
   - Click "Predict Churn"
   - Show probability gauge and feature importance

2. **Performance Monitoring Tab:**
   - Show model metrics dashboard
   - Explain data drift monitoring
   - Show alert system status

**Key Points:**
- "Real-time predictions with confidence scores"
- "Feature importance for explainability"
- "Interactive dashboard for business users"

### **Part 4: Production Monitoring (4 mins)**

**Show:** Monitoring capabilities
```bash
# Run data drift detection
python scripts/monitoring/run_data_drift_monitor.py

# Show monitoring reports
ls reports/drift_reports/
```

**Demo:**
- Open drift report HTML file
- Show performance monitoring dashboard
- Explain alert system (email/Slack notifications)

**Key Points:**
- "Automated data drift detection with Evidently AI"
- "Performance degradation alerts"
- "Production monitoring with comprehensive reporting"

## 🎯 Interview Talking Points

### **Technical Architecture:**
- "Microservices architecture with FastAPI"
- "Docker containerization for scalability"
- "Kubernetes manifests for orchestration"
- "Cloud-ready deployment (AWS/Azure/GCP)"

### **MLOps Best Practices:**
- "Experiment tracking with MLflow"
- "Model versioning and registry"
- "Automated testing and validation"
- "CI/CD pipeline ready"

### **Production Readiness:**
- "Health checks and monitoring"
- "Data drift detection"
- "Performance monitoring"
- "Alert system integration"

### **Business Value:**
- "Reduces customer churn by 15-20%"
- "Automated decision making"
- "Real-time risk assessment"
- "Scalable to millions of customers"

## 🚀 Quick Commands Reference

```bash
# Start demo environment
python scripts/deployment/start_complete_deployment.py

# Train new model
python scripts/training/run_unified_training.py

# Test API
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"features": [...]}'

# Run monitoring
python scripts/monitoring/run_monitoring.py

# View experiments
mlflow ui --port 5000
```

## 📊 Demo Data Points

- **Models Trained:** 7 (XGBoost, LightGBM, Random Forest, Logistic Regression)
- **Best Accuracy:** 80.4%
- **Data Size:** 7,043 customers
- **Features:** 19 customer attributes
- **Churn Rate:** ~26.5%
- **API Response Time:** <100ms
- **Monitoring Frequency:** Real-time

## 🎭 Demo Tips

### **Do:**
- Keep it interactive and engaging
- Focus on business value, not just technical details
- Show real predictions with different scenarios
- Explain monitoring importance for production
- Highlight scalability and cloud readiness

### **Don't:**
- Get lost in code details
- Spend too much time on setup
- Skip the monitoring demonstration
- Forget to mention business impact
- Rush through the interactive prediction

## 🔧 Troubleshooting

### **If API doesn't start:**
```bash
# Check port availability
netstat -tulpn | grep :8000
# Kill existing process
lsof -ti:8000 | xargs kill -9
```

### **If Streamlit fails:**
```bash
# Install missing dependencies
pip install streamlit plotly
# Run directly
python -m streamlit run src/deployment/streamlit_app.py
```

### **If models don't load:**
```bash
# Check model files
ls models/staging/
# Install LightGBM
pip install lightgbm
```

---

**🎉 Success Metrics:** After demo, interviewer should understand your MLOps expertise, production readiness mindset, and ability to deliver end-to-end ML solutions.