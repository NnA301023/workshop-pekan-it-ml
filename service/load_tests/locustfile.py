#!/usr/bin/env python3
"""
Load testing script for Iris ML Service using Locust
"""

import json
import random
from typing import Dict
from locust import HttpUser, task, between, events

class IrisMLServiceUser(HttpUser):
    """Locust user class for testing Iris ML Service"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a user starts"""
        print("Starting load test...")
    
    def on_stop(self):
        """Called when a user stops"""
        print("Stopping load test...")
    
    @task(3)  # Higher weight - more frequent
    def test_health_endpoint(self):
        """Test health endpoint"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed with status {response.status_code}")
    
    @task(2)
    def test_root_endpoint(self):
        """Test root endpoint"""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "message" in data and "version" in data:
                    response.success()
                else:
                    response.failure("Root endpoint missing required fields")
            else:
                response.failure(f"Root endpoint failed with status {response.status_code}")
    
    @task(1)
    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        with self.client.get("/model_info", catch_response=True) as response:
            if response.status_code in [200, 503]:  # 503 is expected if model not loaded
                response.success()
            else:
                response.failure(f"Model info failed with status {response.status_code}")
    
    @task(4)  # Highest weight - most frequent
    def test_single_prediction(self):
        """Test single prediction endpoint"""
        # Generate random Iris features
        features = self._generate_random_features()
        
        with self.client.post("/predict", 
                            json=features, 
                            headers={"Content-Type": "application/json"},
                            catch_response=True) as response:
            if response.status_code in [200, 503]:  # 503 is expected if model not loaded
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if "prediction" in data and "probability" in data:
                            response.success()
                        else:
                            response.failure("Prediction response missing required fields")
                    except json.JSONDecodeError:
                        response.failure("Invalid JSON response")
                else:
                    response.success()  # 503 is expected when model not loaded
            else:
                response.failure(f"Prediction failed with status {response.status_code}")
    
    @task(2)
    def test_batch_prediction(self):
        """Test batch prediction endpoint"""
        # Generate multiple random Iris features
        features_list = [self._generate_random_features() for _ in range(random.randint(1, 5))]
        
        with self.client.post("/predict_batch", 
                            json=features_list,
                            headers={"Content-Type": "application/json"},
                            catch_response=True) as response:
            if response.status_code in [200, 503]:  # 503 is expected if model not loaded
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if "predictions" in data and len(data["predictions"]) == len(features_list):
                            response.success()
                        else:
                            response.failure("Batch prediction response missing required fields")
                    except json.JSONDecodeError:
                        response.failure("Invalid JSON response")
                else:
                    response.success()  # 503 is expected when model not loaded
            else:
                response.failure(f"Batch prediction failed with status {response.status_code}")
    
    def _generate_random_features(self) -> Dict[str, float]:
        """Generate random Iris features within realistic ranges"""
        return {
            "sepal_length": round(random.uniform(4.0, 8.0), 2),
            "sepal_width": round(random.uniform(2.0, 4.5), 2),
            "petal_length": round(random.uniform(1.0, 7.0), 2),
            "petal_width": round(random.uniform(0.1, 2.5), 2)
        }

class IrisMLServiceLoadTest(HttpUser):
    """Alternative user class for more intensive load testing"""
    
    wait_time = between(0.5, 1.5)  # Faster requests
    
    @task(5)
    def test_rapid_predictions(self):
        """Test rapid single predictions"""
        features = {
            "sepal_length": round(random.uniform(4.0, 8.0), 2),
            "sepal_width": round(random.uniform(2.0, 4.5), 2),
            "petal_length": round(random.uniform(1.0, 7.0), 2),
            "petal_width": round(random.uniform(0.1, 2.5), 2)
        }
        
        with self.client.post("/predict", 
                            json=features,
                            headers={"Content-Type": "application/json"},
                            catch_response=True) as response:
            if response.status_code in [200, 503]:
                response.success()
            else:
                response.failure(f"Rapid prediction failed with status {response.status_code}")
    
    @task(2)
    def test_large_batch_predictions(self):
        """Test large batch predictions"""
        features_list = [{
            "sepal_length": round(random.uniform(4.0, 8.0), 2),
            "sepal_width": round(random.uniform(2.0, 4.5), 2),
            "petal_length": round(random.uniform(1.0, 7.0), 2),
            "petal_width": round(random.uniform(0.1, 2.5), 2)
        } for _ in range(random.randint(10, 20))]
        
        with self.client.post("/predict_batch", 
                            json=features_list,
                            headers={"Content-Type": "application/json"},
                            catch_response=True) as response:
            if response.status_code in [200, 503]:
                response.success()
            else:
                response.failure(f"Large batch prediction failed with status {response.status_code}")

# Event handlers for additional monitoring
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the test starts"""
    print("Starting Iris ML Service load test...")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when the test stops"""
    print("Stopping Iris ML Service load test...")

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, 
               context, exception, start_time, url, **kwargs):
    """Called for every request"""
    if exception:
        print(f"Request failed: {name} - {exception}")
    elif response.status_code >= 400:
        print(f"Request error: {name} - Status {response.status_code}")
