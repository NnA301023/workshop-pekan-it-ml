#!/bin/bash

# Iris ML Service Setup Script
# This script sets up and runs the complete ML service

set -e

echo "🚀 Starting Iris ML Service Setup..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose and try again."
    exit 1
fi

print_status "Docker and Docker Compose are available"

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p models logs

# Build the Docker image
print_status "Building Docker image..."
docker-compose build

# Train the model
print_status "Training the model..."
docker-compose --profile training up model-trainer

# Check if model was created
if [ ! -f "models/iris_model.joblib" ]; then
    print_error "Model training failed. Model file not found."
    exit 1
fi

print_status "Model training completed successfully!"

# Start the service
print_status "Starting the ML service..."
docker-compose up -d iris-ml-service

# Wait for service to be ready
print_status "Waiting for service to be ready..."
sleep 10

# Test the service
print_status "Testing the service..."

# Test health endpoint
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    print_status "Health check passed!"
else
    print_warning "Health check failed. Service might still be starting up."
fi

# Test prediction endpoint
TEST_RESPONSE=$(curl -s -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }' 2>/dev/null || echo "{}")

if echo "$TEST_RESPONSE" | grep -q "prediction"; then
    print_status "Prediction endpoint is working!"
    echo "Test prediction response: $TEST_RESPONSE"
else
    print_warning "Prediction endpoint test failed. Service might still be starting up."
fi

print_status "Setup completed!"
echo ""
echo "📊 Service Information:"
echo "   - API Documentation: http://localhost:8000/docs"
echo "   - Health Check: http://localhost:8000/health"
echo "   - Service URL: http://localhost:8000"
echo ""
echo "🧪 To run load tests:"
echo "   docker-compose --profile load-testing up locust"
echo "   Then visit: http://localhost:8089"
echo ""
echo "📝 To view logs:"
echo "   docker-compose logs -f iris-ml-service"
echo ""
echo "🛑 To stop the service:"
echo "   docker-compose down"
