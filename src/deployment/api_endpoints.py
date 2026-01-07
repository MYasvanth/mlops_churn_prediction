"""
FastAPI Endpoints for Churn Prediction Model Serving
Separates API layer from model serving logic for better modularity
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from datetime import datetime

from ..models.unified_model_registry_fixed import UnifiedModelRegistry
from ..utils.logger import get_logger
from ..utils.config_loader import load_config

# Configure logging
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    version="1.0.0",
    description="REST API for churn prediction model serving",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global registry instance
registry = None

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    features: List[float]
    model_id: Optional[str] = None
    stage: Optional[str] = "production"

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_id: str
    timestamp: str

class BatchPredictionRequest(BaseModel):
    data: List[List[float]]
    model_id: Optional[str] = None
    stage: Optional[str] = "production"

class BatchPredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]
    model_id: str
    timestamp: str

class ModelInfo(BaseModel):
    model_id: str
    model_type: str
    stage: str
    version: str
    features: List[str]
    registered_at: str
    performance_metrics: Dict[str, float]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    available_models: int
    timestamp: str

def get_registry():
    """Dependency to get the model registry instance."""
    global registry
    if registry is None:
        try:
            registry = UnifiedModelRegistry()
            logger.info("Model registry initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model registry: {e}")
            raise HTTPException(status_code=500, detail="Model registry initialization failed")
    return registry

@app.on_event("startup")
async def startup_event():
    """Initialize model registry on startup."""
    try:
        global registry
        registry = UnifiedModelRegistry()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Startup error: {e}")

@app.get("/health", response_model=HealthResponse)
async def health_check(registry: UnifiedModelRegistry = Depends(get_registry)):
    """Health check endpoint."""
    try:
        models = registry.list_models()
        return HealthResponse(
            status="healthy",
            model_loaded=len(models) > 0,
            available_models=len(models),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.get("/models", response_model=List[ModelInfo])
async def list_models(
    stage: Optional[str] = None,
    registry: UnifiedModelRegistry = Depends(get_registry)
):
    """List all available models."""
    try:
        models = registry.list_models(stage)
        return [
            ModelInfo(
                model_id=model["model_id"],
                model_type=model.get("model_type", "unknown"),
                stage=model.get("stage", "unknown"),
                version=model.get("version", "1.0.0"),
                features=model.get("feature_columns", []),
                registered_at=model.get("registered_at", ""),
                performance_metrics=model.get("performance_metrics", {})
            )
            for model in models
        ]
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve models")

@app.get("/models/{model_id}", response_model=ModelInfo)
async def get_model_info(
    model_id: str,
    stage: str = "production",
    registry: UnifiedModelRegistry = Depends(get_registry)
):
    """Get detailed information about a specific model."""
    try:
        model_info = registry.get_model_info(model_id, stage)
        return ModelInfo(
            model_id=model_info["model_id"],
            model_type=model_info.get("model_type", "unknown"),
            stage=model_info.get("stage", "unknown"),
            version=model_info.get("version", "1.0.0"),
            features=model_info.get("feature_columns", []),
            registered_at=model_info.get("registered_at", ""),
            performance_metrics=model_info.get("performance_metrics", {})
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    registry: UnifiedModelRegistry = Depends(get_registry)
):
    """Single prediction endpoint."""
    try:
        # Load the specified model or default to latest production
        if request.model_id:
            model = registry.load_model(request.model_id, request.stage)
        else:
            # Get latest production model
            models = registry.list_models("production")
            if not models:
                raise HTTPException(status_code=404, detail="No production models available")
            latest_model = models[0]
            model = registry.load_model(latest_model["model_id"], "production")
            request.model_id = latest_model["model_id"]
        
        # Convert features to numpy array
        features = np.array(request.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get probability
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(features)[0][1]
        else:
            probability = float(prediction)
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            model_id=request.model_id,
            timestamp=datetime.now().isoformat()
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    registry: UnifiedModelRegistry = Depends(get_registry)
):
    """Batch prediction endpoint."""
    try:
        # Load the specified model or default to latest production
        if request.model_id:
            model = registry.load_model(request.model_id, request.stage)
        else:
            # Get latest production model
            models = registry.list_models("production")
            if not models:
                raise HTTPException(status_code=404, detail="No production models available")
            latest_model = models[0]
            model = registry.load_model(latest_model["model_id"], "production")
            request.model_id = latest_model["model_id"]
        
        # Convert to numpy array
        features = np.array(request.data)
        
        # Make predictions
        predictions = model.predict(features)
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[:, 1]
        else:
            probabilities = predictions.astype(float)
        
        return BatchPredictionResponse(
            predictions=predictions.tolist(),
            probabilities=probabilities.tolist(),
            model_id=request.model_id,
            timestamp=datetime.now().isoformat()
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API documentation."""
    return {
        "message": "Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "list_models": "/models",
            "model_info": "/models/{model_id}",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "documentation": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.deployment.api_endpoints:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
