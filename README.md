# MLOps Churn Prediction Pipeline

An end-to-end MLOps pipeline for customer churn prediction with automated training, deployment, monitoring, and production-ready serving capabilities.

## 🚀 Live Demo

Streamlit App (Deployed):

👉 https://mlops-churn-prediction-4qvb.onrender.com

This UI loads the production model from MLflow and performs real-time churn prediction.

## 🎯 Overview

This project implements a complete machine learning operations (MLOps) pipeline for predicting customer churn using multiple algorithms and industry best practices. The pipeline includes data processing, model training, hyperparameter optimization, model deployment, and continuous monitoring.

### Key Features

- **Multi-Model Support**: XGBoost, LightGBM, Random Forest, Logistic Regression, SVM, and Ensemble methods
- **Experiment Tracking**: MLflow integration for comprehensive experiment management
- **Pipeline Orchestration**: ZenML for reproducible and scalable ML pipelines
- **Hyperparameter Optimization**: Optuna integration for automated tuning
- **Model Deployment**: FastAPI-based REST API with automatic model serving
- **Interactive Dashboard**: Streamlit application for real-time predictions and monitoring
- **Data Drift Monitoring**: Evidently AI integration for production monitoring
- **Containerization**: Docker and Kubernetes manifests for cloud deployment
- **CI/CD Ready**: Automated testing, validation, and deployment pipelines

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Ingestion│    │  Feature        │    │   Model         │
│   & Validation  │───▶│  Engineering    │───▶│   Training      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Model         │    │   API Serving   │    │   Monitoring    │
│   Registry      │    │   (FastAPI)     │    │   & Alerts      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

The following diagram illustrates the end-to-end MLOps architecture for the churn prediction system.
It demonstrates how data flows from ingestion to deployment while integrating experiment tracking, CI/CD, monitoring, and scalable infrastructure.


![Churn Architecture](https://github.com/MYasvanth/mlops_churn_prediction/blob/master/docs/Churn_Architecture.png)

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| XGBoost | 80.4% | 0.82 | 0.79 | 0.80 |
| LightGBM | 80.1% | 0.81 | 0.78 | 0.79 |
| Random Forest | 79.8% | 0.80 | 0.77 | 0.78 |
| Logistic Regression | 79.2% | 0.79 | 0.76 | 0.77 |

🔬 Evidence-Based Model Selection
Unlike traditional pipelines that deploy models based on a single accuracy score, this project implements a Statistical Validation Layer to ensure every deployment decision is mathematically sound and business-justified:

Reliability with Bootstrap CIs: We calculate 95% Confidence Intervals for all metrics. This ensures we understand the model's performance range (e.g., 0.79±0.010.79 \pm 0.010.79±0.01) rather than relying on a single "lucky" test run.

Hypothesis Testing (Scientific Rigor): We use Paired T-Tests and Wilcoxon tests with Bonferroni Correction to verify that performance gains are statistically significant and not due to random variation.

Practical Significance Threshold: A model is only considered a "winner" if it exceeds a 1% absolute improvement threshold. This ensures that engineering resources are only spent on deployments that provide meaningful business ROI.

Effect Size Analysis: By calculating Cohen's d, we quantify the magnitude of the performance difference, allowing stakeholders to distinguish between "negligible" and "substantial" improvements.

Key Benefit: This framework eliminates "Analysis Paralysis" by converting complex p-values into clear, actionable deployment recommendations (e.g., "Deploy XGBoost: It shows a statistically significant and 2.4% practical improvement over the baseline").



### Running the Pipeline

1. **Train models**
   ```bash
   python run_churn_pipeline.py --model-type xgboost --hyperparameter-optimization
   ```

2. **Start the API server**
   ```bash
   python scripts/monitoring/run_fastapi_server.py
   ```

3. **Launch the dashboard**
   ```bash
   streamlit run src/deployment/streamlit_app.py --server.port 8501
   ```

4. **View experiment tracking**
   ```bash
   mlflow ui --port 5000
   ```

## 📖 Usage

### Training Pipeline

```bash
# Basic training
python run_churn_pipeline.py --model-type xgboost

# With hyperparameter optimization
python run_churn_pipeline.py --model-type lightgbm --hyperparameter-optimization --n-trials 100

# Deploy to production
python run_churn_pipeline.py --model-type xgboost --deploy-to-production
```

### API Endpoints

The FastAPI server provides the following endpoints:

- `GET /health` - Health check
- `GET /models` - List available models
- `POST /predict` - Make predictions
- `GET /monitoring/metrics` - Get monitoring metrics

### Example Prediction Request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "gender": "Female",
       "SeniorCitizen": 0,
       "Partner": "Yes",
       "Dependents": "No",
       "tenure": 12,
       "PhoneService": "Yes",
       "MultipleLines": "No",
       "InternetService": "Fiber optic",
       "OnlineSecurity": "No",
       "OnlineBackup": "Yes",
       "DeviceProtection": "No",
       "TechSupport": "No",
       "StreamingTV": "Yes",
       "StreamingMovies": "No",
       "Contract": "Month-to-month",
       "PaperlessBilling": "Yes",
       "PaymentMethod": "Electronic check",
       "MonthlyCharges": 85.5,
       "TotalCharges": 1023.75
     }'
```

## 🏛️ Project Structure

```
mlops_churn_prediction/
├── configs/                 # Configuration files
│   ├── data/               # Data processing configs
│   ├── deployment/         # Deployment configurations
│   ├── model/              # Model hyperparameters
│   └── monitoring/         # Monitoring settings
├── data/                   # Data directory
│   ├── raw/               # Raw data
│   ├── processed/         # Processed data
│   └── external/          # External data sources
├── deployment/             # Deployment configurations
│   ├── docker/            # Docker files
│   ├── kubernetes/        # K8s manifests
│   └── cloud/             # Cloud deployment configs
├── models/                 # Trained models
│   ├── staging/           # Staging models
│   └── production/        # Production models
├── monitoring/             # Monitoring components
│   ├── alerts/            # Alert configurations
│   ├── dashboards/        # Monitoring dashboards
│   └── logs/              # Application logs
├── notebooks/              # Jupyter notebooks
├── reports/                # Generated reports
│   ├── drift_reports/     # Data drift reports
│   └── performance_reports/ # Model performance reports
├── scripts/                # Utility scripts
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── features/          # Feature engineering
│   ├── models/            # Model training modules
│   ├── deployment/        # Deployment modules
│   ├── monitoring/        # Monitoring modules
│   └── pipelines/         # ML pipelines
├── tests/                  # Test suites
├── zenml_pipelines/        # ZenML pipeline definitions
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup
├── dvc.yaml              # Data version control
└── README.md             # Project documentation
```


## 📈 Monitoring & Observability

### Data Drift Detection

```bash
# Run drift detection
python scripts/monitoring/run_data_drift_monitor.py

# View drift reports
ls reports/drift_reports/
```

### Performance Monitoring

- **MLflow UI**: http://localhost:5000
- **Streamlit Dashboard**: http://localhost:8501
- **Optuna Dashboard**: http://localhost:8080
- **API Documentation**: http://localhost:8000/docs

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_models/

# Run with coverage
pytest --cov=src --cov-report=html
```


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

