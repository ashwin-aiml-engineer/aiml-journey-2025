# FastAPI Fundamentals for ML
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Basic FastAPI app
app = FastAPI(title="ML API", description="Machine Learning Service", version="1.0.0")

# Request/Response models with Pydantic
class PredictionRequest(BaseModel):
    features: list
    
class PredictionResponse(BaseModel):
    prediction: float
    confidence: float

# ML Model API Integration
# Load a pre-trained model (simulation)
model = LinearRegression()
X_train = np.random.rand(100, 4)
y_train = np.random.rand(100)
model.fit(X_train, y_train)

@app.get("/")
async def root():
    return {"message": "ML API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Input validation and preprocessing
        if len(request.features) != 4:
            raise HTTPException(status_code=400, detail="Expected 4 features")
        
        # Model inference
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        confidence = 0.95  # Simulated confidence
        
        return PredictionResponse(prediction=prediction, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Docker Containerization Basics
dockerfile_content = """
# Dockerfile for ML API
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# Requirements.txt content
requirements_content = """
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
scikit-learn==1.3.2
numpy==1.24.3
pandas==2.0.3
joblib==1.3.2
"""

# Docker Compose for multi-service applications
docker_compose_content = """
version: '3.8'
services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models
    volumes:
      - ./models:/app/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
"""

# Client-facing Application with Gradio
import gradio as gr

def gradio_predict(feature1, feature2, feature3, feature4):
    """Gradio interface for ML model"""
    features = np.array([feature1, feature2, feature3, feature4]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return f"Prediction: {prediction:.3f}"

# Gradio interface
interface = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Number(label="Feature 1"),
        gr.Number(label="Feature 2"),
        gr.Number(label="Feature 3"),
        gr.Number(label="Feature 4")
    ],
    outputs=gr.Text(label="Result"),
    title="ML Model Demo",
    description="Interactive ML model prediction"
)

# Production Readiness - Logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response

# Model versioning
@app.get("/model/version")
async def get_model_version():
    return {"model_version": "1.0.0", "api_version": "1.0.0"}

if __name__ == "__main__":
    print("FastAPI ML Service Setup Complete!")
    print("\nTo run:")
    print("1. Install: pip install fastapi uvicorn gradio")
    print("2. Run API: uvicorn main:app --reload")
    print("3. Access docs: http://localhost:8000/docs")
    print("4. Run Gradio: interface.launch()")
    
    # Save Docker files
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
    
    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose_content)
    
    print("\nDocker files created: Dockerfile, requirements.txt, docker-compose.yml")
    print("To build: docker build -t ml-api .")
    print("To run: docker-compose up")