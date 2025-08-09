#!/usr/bin/env python3
"""
FastAPI application for Iris ML service
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Iris ML Service",
    description="A machine learning service for Iris flower classification using Random Forest",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and metadata
model = None
metadata = None

class IrisFeatures(BaseModel):
    """Pydantic model for Iris features"""
    sepal_length: float = Field(..., ge=0, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, description="Petal length in cm")
    petal_width: float = Field(..., ge=0, description="Petal width in cm")

class PredictionResponse(BaseModel):
    """Pydantic model for prediction response"""
    prediction: str
    probability: float
    features: Dict[str, float]

class HealthResponse(BaseModel):
    """Pydantic model for health check response"""
    status: str
    model_loaded: bool
    model_info: Optional[Dict[str, Any]] = None

def load_model():
    """Load the trained model and metadata"""
    global model, metadata
    
    try:
        model_path = Path("models/iris_model.joblib")
        metadata_path = Path("models/model_metadata.json")
        
        if not model_path.exists():
            logger.error("Model file not found. Please train the model first.")
            return False
        
        # Load model
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        
        # Load metadata
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info("Metadata loaded successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting Iris ML Service...")
    load_model()

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Iris ML Service",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_info = None
    if metadata:
        model_info = {
            "model_type": metadata.get("model_type"),
            "feature_names": metadata.get("feature_names"),
            "target_names": metadata.get("target_names")
        }
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_info=model_info
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures):
    """Predict Iris species"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to numpy array
        feature_values = [
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]
        
        X = np.array([feature_values])
        
        # Make prediction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Get predicted class name
        if metadata and "target_names" in metadata:
            predicted_class = metadata["target_names"][prediction]
        else:
            predicted_class = f"class_{prediction}"
        
        # Get probability for predicted class
        probability = float(probabilities[prediction])
        
        # Convert features to dict
        feature_dict = {
            "sepal_length": features.sepal_length,
            "sepal_width": features.sepal_width,
            "petal_length": features.petal_length,
            "petal_width": features.petal_width
        }
        
        logger.info(f"Prediction made: {predicted_class} (confidence: {probability:.4f})")
        
        return PredictionResponse(
            prediction=predicted_class,
            probability=probability,
            features=feature_dict
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(features_list: List[IrisFeatures]):
    """Predict multiple Iris samples"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to numpy array
        X = []
        for features in features_list:
            X.append([
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width
            ])
        
        X = np.array(X)
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Format results
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            if metadata and "target_names" in metadata:
                predicted_class = metadata["target_names"][pred]
            else:
                predicted_class = f"class_{pred}"
            
            results.append({
                "sample_id": i,
                "prediction": predicted_class,
                "probability": float(prob[pred]),
                "features": {
                    "sepal_length": features_list[i].sepal_length,
                    "sepal_width": features_list[i].sepal_width,
                    "petal_length": features_list[i].petal_length,
                    "petal_width": features_list[i].petal_width
                }
            })
        
        logger.info(f"Batch prediction completed: {len(results)} samples")
        
        return {"predictions": results}
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """Get model information"""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not available")
    
    return {
        "model_type": metadata.get("model_type"),
        "feature_names": metadata.get("feature_names"),
        "target_names": metadata.get("target_names"),
        "model_parameters": {
            "n_estimators": metadata.get("n_estimators"),
            "max_depth": metadata.get("max_depth")
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
