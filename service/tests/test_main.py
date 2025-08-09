#!/usr/bin/env python3
"""
Unit tests for the Iris ML Service FastAPI application
"""

import pytest
import json
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import the app
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))
from main import app

client = TestClient(app)

class TestHealthEndpoints:
    """Test health and status endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["message"] == "Iris ML Service"
    
    def test_health_endpoint(self):
        """Test health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)

class TestModelInfo:
    """Test model information endpoints"""
    
    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        response = client.get("/model_info")
        # This might fail if model is not loaded, which is expected
        # We'll test the error case
        if response.status_code == 503:
            assert "Model metadata not available" in response.json()["detail"]
        else:
            assert response.status_code == 200
            data = response.json()
            assert "model_type" in data
            assert "feature_names" in data
            assert "target_names" in data

class TestPredictionEndpoints:
    """Test prediction endpoints"""
    
    def test_predict_valid_input(self):
        """Test prediction with valid input"""
        test_features = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = client.post("/predict", json=test_features)
        
        if response.status_code == 503:
            # Model not loaded, which is expected in test environment
            assert "Model not loaded" in response.json()["detail"]
        else:
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert "probability" in data
            assert "features" in data
            assert data["features"] == test_features
            assert isinstance(data["probability"], float)
            assert 0 <= data["probability"] <= 1
    
    def test_predict_invalid_input(self):
        """Test prediction with invalid input"""
        # Test with negative values
        invalid_features = {
            "sepal_length": -1.0,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = client.post("/predict", json=invalid_features)
        assert response.status_code == 422  # Validation error
    
    def test_predict_missing_fields(self):
        """Test prediction with missing fields"""
        incomplete_features = {
            "sepal_length": 5.1,
            "sepal_width": 3.5
            # Missing petal_length and petal_width
        }
        
        response = client.post("/predict", json=incomplete_features)
        assert response.status_code == 422  # Validation error
    
    def test_predict_batch_valid_input(self):
        """Test batch prediction with valid input"""
        test_features_list = [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            },
            {
                "sepal_length": 6.3,
                "sepal_width": 3.3,
                "petal_length": 4.7,
                "petal_width": 1.6
            }
        ]
        
        response = client.post("/predict_batch", json=test_features_list)
        
        if response.status_code == 503:
            # Model not loaded, which is expected in test environment
            assert "Model not loaded" in response.json()["detail"]
        else:
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 2
            
            for i, prediction in enumerate(data["predictions"]):
                assert "sample_id" in prediction
                assert "prediction" in prediction
                assert "probability" in prediction
                assert "features" in prediction
                assert prediction["sample_id"] == i
                assert prediction["features"] == test_features_list[i]
    
    def test_predict_batch_invalid_input(self):
        """Test batch prediction with invalid input"""
        invalid_features_list = [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            },
            {
                "sepal_length": -1.0,  # Invalid negative value
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        ]
        
        response = client.post("/predict_batch", json=invalid_features_list)
        assert response.status_code == 422  # Validation error

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_predict_with_zero_values(self):
        """Test prediction with zero values (should be valid)"""
        zero_features = {
            "sepal_length": 0.0,
            "sepal_width": 0.0,
            "petal_length": 0.0,
            "petal_width": 0.0
        }
        
        response = client.post("/predict", json=zero_features)
        
        if response.status_code == 503:
            # Model not loaded, which is expected in test environment
            assert "Model not loaded" in response.json()["detail"]
        else:
            assert response.status_code == 200
    
    def test_predict_with_large_values(self):
        """Test prediction with large values"""
        large_features = {
            "sepal_length": 100.0,
            "sepal_width": 100.0,
            "petal_length": 100.0,
            "petal_width": 100.0
        }
        
        response = client.post("/predict", json=large_features)
        
        if response.status_code == 503:
            # Model not loaded, which is expected in test environment
            assert "Model not loaded" in response.json()["detail"]
        else:
            assert response.status_code == 200
    
    def test_predict_with_decimal_values(self):
        """Test prediction with decimal values"""
        decimal_features = {
            "sepal_length": 5.123456,
            "sepal_width": 3.456789,
            "petal_length": 1.789012,
            "petal_width": 0.234567
        }
        
        response = client.post("/predict", json=decimal_features)
        
        if response.status_code == 503:
            # Model not loaded, which is expected in test environment
            assert "Model not loaded" in response.json()["detail"]
        else:
            assert response.status_code == 200

class TestModelLoading:
    """Test model loading functionality"""
    
    @patch('main.joblib.load')
    @patch('main.Path.exists')
    def test_model_loading_success(self, mock_exists, mock_load):
        """Test successful model loading"""
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        # Import and test the load_model function
        from main import load_model
        result = load_model()
        assert result is True
    
    @patch('main.Path.exists')
    def test_model_loading_failure(self, mock_exists):
        """Test model loading failure when file doesn't exist"""
        mock_exists.return_value = False
        
        from main import load_model
        result = load_model()
        assert result is False

if __name__ == "__main__":
    pytest.main([__file__])
