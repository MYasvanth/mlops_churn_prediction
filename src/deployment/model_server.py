
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Churn Prediction API", version="1.0.0")

# Global variables for model and preprocessor
model = None
preprocessor = None
feature_names = None

class PredictionRequest(BaseModel):
    features: List[float]
    
class PredictionResponse(BaseModel):
    prediction: int
    probability: float

class BatchPredictionRequest(BaseModel):
    data: List[List[float]]

class ModelInfo(BaseModel):
    model_type: str
    version: str
    features: List[str]
def load_model_and_preprocessor():
    """Load the trained model and preprocessor."""
    global model, preprocessor, feature_names
    
    try:
        # Try loading from local files first
        model_path = Path("models/production/model.pkl")
        preprocessor_path = Path("models/production/preprocessor.pkl")
        
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            # Try loading from MLflow local path
            mlflow_model_path = Path("./mlruns")
            if mlflow_model_path.exists():
                # Find the latest model
                runs = list(mlflow_model_path.glob("*/"))
                if runs:
                    latest_run = max(runs, key=lambda x: x.stat().st_mtime)
                    model_file = latest_run / "artifacts/model/model.pkl"
                    if model_file.exists():
                        model = joblib.load(model_file)
                        logger.info(f"Model loaded from {model_file}")
                    else:
                        # Try loading from artifacts
                        model_file = latest_run / "artifacts/model.pkl"
                        if model_file.exists():
                            model = joblib.load(model_file)
                            logger.info(f"Model loaded from {model_file}")
                        else:
                            raise FileNotFoundError("No model file found")
            else:
                raise FileNotFoundError("No model files found")
            
        if preprocessor_path.exists():
            preprocessor = joblib.load(preprocessor_path)
            logger.info(f"Preprocessor loaded from {preprocessor_path}")
            
        # Set feature names based on training data
        feature_names = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges"
        ]
        
        # Log model state
        if model is not None:
            logger.info("Model is ready for predictions.")
        else:
            logger.error("Model is not loaded properly.")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Create a simple fallback model for testing
        from sklearn.dummy import DummyClassifier
        model = DummyClassifier(strategy="most_frequent")
        model.fit([[0] * 19], [0])  # Dummy fit
        logger.warning("Using fallback dummy model")




@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model_and_preprocessor()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_type=type(model).__name__,
        version="1.0.0",
        features=feature_names
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to numpy array
        features = np.array(request.features).reshape(1, -1)
        
        # Apply preprocessing if available
        if preprocessor is not None:
            features = preprocessor.transform(features)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get probability (handle models without predict_proba)
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(features)[0][1]
        else:
            probability = float(prediction)
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to numpy array
        features = np.array(request.data)
        
        # Apply preprocessing if available
        if preprocessor is not None:
            features = preprocessor.transform(features)
        
        # Make predictions
        predictions = model.predict(features)
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[:, 1]
        else:
            probabilities = predictions.astype(float)
        
        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Churn Prediction API",
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "predict": "/predict",
            "predict_batch": "/predict/batch"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "src.deployment.model_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
