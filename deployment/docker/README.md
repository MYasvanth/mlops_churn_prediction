# Containerized Execution for MLOps Churn Prediction

This directory contains the Docker configuration for running the MLOps churn prediction project in a containerized environment.

## Files

### Dockerfile
The main Dockerfile that defines the container environment:
- Base image: Python 3.8 slim
- Installs all dependencies from requirements.txt
- Copies the entire project into the container
- Exposes ports for MLflow (5000), FastAPI (8000), Streamlit (8501), and Optuna (8080)
- Default command: Runs the complete deployment script

### docker-compose.yml
Docker Compose configuration for orchestrating multiple services:
- **mlflow**: MLflow tracking server
- **fastapi**: FastAPI model serving
- **streamlit**: Streamlit dashboard
- **optuna**: Optuna dashboard for hyperparameter optimization

## Usage

### 1. Build the Docker Image
```bash
docker build -t mlops-churn-prediction -f deployment/docker/Dockerfile .
```

### 2. Run Containerized Training
```bash
docker run \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/mlartifacts:/app/mlartifacts \
  mlops-churn-prediction \
  python scripts/run_training.py
```

### 3. Run Containerized ZenML Pipeline
```bash
docker run \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/mlartifacts:/app/mlartifacts \
  -v $(pwd)/zenml_pipelines:/app/zenml_pipelines \
  mlops-churn-prediction \
  python run_churn_pipeline.py --model-type xgboost --deploy-to-production
```

### 4. Full Docker Compose Deployment
```bash
docker-compose -f deployment/docker/docker-compose.yml up -d
```

### 5. Using the Containerized Execution Script
```bash
python scripts/containerized_execution.py
```

## Volume Mounts

The container uses the following volume mounts to persist data:
- `/app/data` → Host `data/` directory
- `/app/models` → Host `models/` directory  
- `/app/mlartifacts` → Host `mlartifacts/` directory
- `/app/zenml_pipelines` → Host `zenml_pipelines/` directory

## Ports

- **5000**: MLflow tracking UI
- **8000**: FastAPI model serving
- **8501**: Streamlit dashboard
- **8080**: Optuna dashboard

## Benefits of Containerized Execution

1. **Reproducibility**: Consistent environment across all executions
2. **Isolation**: No conflicts with host system dependencies
3. **Scalability**: Easy to deploy to cloud platforms
4. **Portability**: Run anywhere Docker is available
5. **Version Control**: Container images can be versioned and tagged

## Development Workflow

1. Develop and test locally
2. Build Docker image for testing
3. Run containerized execution to verify
4. Push image to container registry for deployment
5. Deploy to production environment

## Production Deployment

For production deployment, consider:
- Using a multi-stage Docker build
- Setting up proper health checks
- Configuring resource limits
- Implementing logging and monitoring
- Using environment variables for configuration
