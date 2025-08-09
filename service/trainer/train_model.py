#!/usr/bin/env python3
"""
Model training script for Iris dataset using Random Forest
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IrisModelTrainer:
    """Trainer class for Iris dataset using Random Forest"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model = None
        self.feature_names = None
        self.target_names = None
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, list, list]:
        """Load Iris dataset"""
        logger.info("Loading Iris dataset...")
        iris = load_iris()
        X = iris.data
        y = iris.target
        feature_names = iris.feature_names
        target_names = iris.target_names
        
        logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Feature names: {feature_names}")
        logger.info(f"Target names: {target_names}")
        
        return X, y, feature_names, target_names
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        """Train Random Forest model"""
        logger.info("Training Random Forest model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize and train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5)
        
        logger.info(f"Model accuracy: {accuracy:.4f}")
        logger.info(f"Cross-validation scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Detailed evaluation
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred, target_names=self.target_names))
        
        return model
    
    def save_model(self, model: RandomForestClassifier, feature_names: list, target_names: list):
        """Save trained model and metadata"""
        logger.info("Saving model and metadata...")
        
        # Save model
        model_path = self.model_dir / "iris_model.joblib"
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'feature_names': feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names),
            'target_names': target_names.tolist() if hasattr(target_names, 'tolist') else list(target_names),
            'model_type': 'RandomForestClassifier',
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth
        }
        
        metadata_path = self.model_dir / "model_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Metadata saved to: {metadata_path}")
    
    def run_training(self):
        """Complete training pipeline"""
        try:
            # Load data
            X, y, feature_names, target_names = self.load_data()
            self.feature_names = feature_names
            self.target_names = target_names
            
            # Train model
            model = self.train_model(X, y)
            self.model = model
            
            # Save model
            self.save_model(model, feature_names, target_names)
            
            logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

def main():
    """Main function to run training"""
    trainer = IrisModelTrainer()
    trainer.run_training()

if __name__ == "__main__":
    main()
