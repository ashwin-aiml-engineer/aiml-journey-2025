from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import uvicorn
import json
from datetime import datetime

# ================================
# PYDANTIC MODELS FOR API
# ================================

class CustomerData(BaseModel):
    """Single customer data for churn prediction"""
    age: int = Field(..., ge=18, le=100, description="Customer age")
    monthly_charges: float = Field(..., ge=0, le=200, description="Monthly charges in USD")
    total_charges: float = Field(..., ge=0, description="Total charges in USD")
    contract_length: int = Field(..., description="Contract length in months (1, 12, or 24)")
    support_calls: int = Field(..., ge=0, le=20, description="Number of support calls")
    
    class Config:
        schema_extra = {
            "example": {
                "age": 45,
                "monthly_charges": 85.0,
                "total_charges": 2550.0,
                "contract_length": 12,
                "support_calls": 3
            }
        }

class BatchCustomerData(BaseModel):
    """Batch of customers for prediction"""
    customers: List[CustomerData]
    
    class Config:
        schema_extra = {
            "example": {
                "customers": [
                    {
                        "age": 25,
                        "monthly_charges": 50.0,
                        "total_charges": 600.0,
                        "contract_length": 24,
                        "support_calls": 1
                    },
                    {
                        "age": 60,
                        "monthly_charges": 95.0,
                        "total_charges": 5700.0,
                        "contract_length": 1,
                        "support_calls": 5
                    }
                ]
            }
        }

class PredictionResponse(BaseModel):
    """Single prediction response"""
    customer_id: str
    prediction: int = Field(..., description="Churn prediction (0=No, 1=Yes)")
    probability_churn: float = Field(..., description="Probability of churning")
    risk_level: str = Field(..., description="Risk level (Low/Medium/High)")
    timestamp: str

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_customers: int
    high_risk_count: int
    timestamp: str

# ================================
# ML MODEL SERVICE CLASS
# ================================

class ChurnPredictionService:
    """ML service for customer churn prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.feature_names = ['age', 'monthly_charges', 'total_charges', 'contract_length', 'support_calls']
        
    def train_model(self):
        """Train the churn prediction model"""
        print("🔄 Training churn prediction model...")
        
        # Generate training data (in production, load from database/file)
        np.random.seed(42)
        n_samples = 1000
        
        # Create synthetic training data
        age = np.random.randint(18, 80, n_samples)
        monthly_charges = np.random.normal(65, 20, n_samples)
        total_charges = monthly_charges * np.random.randint(1, 60, n_samples)
        contract_length = np.random.choice([1, 12, 24], n_samples, p=[0.3, 0.4, 0.3])
        support_calls = np.random.poisson(2, n_samples)
        
        # Create realistic churn labels
        churn_probability = (
            0.3 * (age < 30).astype(int) +
            0.2 * (monthly_charges > 80).astype(int) +
            0.3 * (contract_length == 1).astype(int) +
            0.2 * (support_calls > 3).astype(int)
        )
        churn = (churn_probability + np.random.normal(0, 0.1, n_samples)) > 0.5
        
        # Create DataFrame
        X = pd.DataFrame({
            'age': age,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'contract_length': contract_length,
            'support_calls': support_calls
        })
        y = churn.astype(int)
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        self.is_trained = True
        
        print(f"✅ Model trained successfully!")
        print(f"📊 Train accuracy: {train_score:.3f}")
        print(f"📊 Test accuracy: {test_score:.3f}")
        
        return {"train_accuracy": train_score, "test_accuracy": test_score}
    
    def predict_single(self, customer_data: CustomerData) -> Dict[str, Any]:
        """Make single customer prediction"""
        if not self.is_trained:
            raise HTTPException(status_code=503, detail="Model not trained yet")
        
        try:
            # Convert to DataFrame
            input_data = pd.DataFrame([{
                'age': customer_data.age,
                'monthly_charges': customer_data.monthly_charges,
                'total_charges': customer_data.total_charges,
                'contract_length': customer_data.contract_length,
                'support_calls': customer_data.support_calls
            }])
            
            # Scale features
            input_scaled = self.scaler.transform(input_data)
            
            # Make prediction
            prediction = self.model.predict(input_scaled)[0]
            probability = self.model.predict_proba(input_scaled)[0][1]  # Probability of churn
            
            # Determine risk level
            if probability < 0.3:
                risk_level = "Low"
            elif probability < 0.7:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            return {
                "prediction": int(prediction),
                "probability_churn": float(probability),
                "risk_level": risk_level
            }
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
    
    def predict_batch(self, customers: List[CustomerData]) -> Dict[str, Any]:
        """Make batch predictions"""
        predictions = []
        high_risk_count = 0
        
        for i, customer in enumerate(customers):
            pred_result = self.predict_single(customer)
            
            if pred_result["risk_level"] == "High":
                high_risk_count += 1
                
            predictions.append({
                "customer_id": f"customer_{i+1}",
                **pred_result,
                "timestamp": datetime.now().isoformat()
            })
        
        return {
            "predictions": predictions,
            "total_customers": len(customers),
            "high_risk_count": high_risk_count
        }

# ================================
# FASTAPI APPLICATION
# ================================

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="ML-powered API for predicting customer churn risk",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc
)

# Initialize ML service
ml_service = ChurnPredictionService()

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    print("🚀 Starting Customer Churn Prediction API...")
    ml_service.train_model()
    print("✅ API ready for predictions!")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_trained": ml_service.is_trained,
        "timestamp": datetime.now().isoformat(),
        "service": "churn-prediction"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """Predict churn for single customer"""
    try:
        prediction_result = ml_service.predict_single(customer)
        
        return PredictionResponse(
            customer_id="single_customer",
            prediction=prediction_result["prediction"],
            probability_churn=prediction_result["probability_churn"],
            risk_level=prediction_result["risk_level"],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_churn_batch(batch_data: BatchCustomerData):
    """Predict churn for multiple customers"""
    try:
        batch_results = ml_service.predict_batch(batch_data.customers)
        
        return BatchPredictionResponse(
            predictions=[
                PredictionResponse(
                    customer_id=pred["customer_id"],
                    prediction=pred["prediction"],
                    probability_churn=pred["probability_churn"],
                    risk_level=pred["risk_level"],
                    timestamp=pred["timestamp"]
                )
                for pred in batch_results["predictions"]
            ],
            total_customers=batch_results["total_customers"],
            high_risk_count=batch_results["high_risk_count"],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    """Get model information and statistics"""
    if not ml_service.is_trained:
        raise HTTPException(status_code=503, detail="Model not trained yet")
    
    return {
        "model_type": "RandomForestClassifier",
        "features": ml_service.feature_names,
        "n_features": len(ml_service.feature_names),
        "is_trained": ml_service.is_trained,
        "model_params": ml_service.model.get_params() if ml_service.model else None
    }

# ================================
# TESTING FUNCTIONS
# ================================

def test_api_locally():
    """Test API endpoints locally"""
    print("🧪 TESTING API LOCALLY")
    print("=" * 40)
    
    # Test single prediction
    test_customer = CustomerData(
        age=45,
        monthly_charges=85.0,
        total_charges=2550.0,
        contract_length=1,
        support_calls=4
    )
    
    # This would be the actual API call testing
    print("✅ API structure ready for testing!")
    print("🌐 Run with: uvicorn day_15_ml_api_service:app --reload")
    print("📖 API docs: http://localhost:8000/docs")

if __name__ == "__main__":
    # For local development
    print("🚀 FASTAPI ML SERVICE - DAY 15")
    print("=" * 50)
    
    test_api_locally()
    
    # Run server (uncomment for actual serving)
    print("\n💡 TO START API SERVER:")
    print("uvicorn day_15_ml_api_service:app --reload --host 0.0.0.0 --port 8000")
    print("\n📱 API ENDPOINTS:")
    print("• GET  /health          - Health check")
    print("• POST /predict         - Single prediction")  
    print("• POST /predict/batch   - Batch predictions")
    print("• GET  /docs            - Interactive API docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)