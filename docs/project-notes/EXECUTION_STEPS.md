# Complete Execution Steps for Streamlit + Optuna Deployment

## Step 1: Environment Setup
```bash
# Activate conda environment
conda activate churn_mlops

# Install dependencies
pip install streamlit optuna-dashboard
```

## Step 2: Run Experiment Tracking
```bash
# Start MLflow tracking
mlflow ui --port 5000

# In another terminal, run training with experiment tracking
python scripts/run_training.py --experiment-name churn_prediction
```

## Step 3: Start Streamlit Dashboard
```bash
# Run Streamlit app
streamlit run src/deployment/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

## Step 4: Start Optuna Dashboard
```bash
# Run Optuna dashboard
optuna-dashboard --port 8080 --host 0.0.0.0 sqlite:///optuna_studies.db
```

## Step 5: Test Model Prediction
```bash
# Test model prediction endpoint
python scripts/test_prediction_endpoint.py
```

## Step 6: Monitor and Track
- **Streamlit**: http://localhost:8501
- **MLflow**: http://localhost:5000
- **Optuna**: http://localhost:8080

## Step 7: Production Deployment
```bash
# Build Docker image
docker build -t churn-prediction-streamlit .

# Run container
docker run -p 8501:8501 churn-prediction-streamlit
