"""
Day 17: Production Model Serving (15-Minute Daily Practice)
🎯 Master FastAPI, Docker & health checks quickly
✅ Essential serving for production ML
"""

import numpy as np
import joblib
from datetime import datetime
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Simple model serving class
class QuickModelServing:
    """Lightweight model serving system"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metrics = {'requests': 0, 'errors': 0}
        self.start_time = datetime.now()
    
    def train_and_save(self):
        """Train and save model quickly"""
        print("🔄 Training model...")
        
        X, y = make_classification(n_samples=500, n_features=5, random_state=42)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.model.fit(X_scaled, y)
        
        # Save to disk
        joblib.dump(self.model, 'quick_model.pkl')
        joblib.dump(self.scaler, 'quick_scaler.pkl')
        print("✅ Model trained and saved")
    
    def load_model(self):
        """Load model from disk"""
        try:
            self.model = joblib.load('quick_model.pkl')
            self.scaler = joblib.load('quick_scaler.pkl')
            print("✅ Model loaded")
            return True
        except FileNotFoundError:
            print("⚠️ Model not found, training new one...")
            self.train_and_save()
            return True
    
    def predict(self, features):
        """Make prediction"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not loaded")
        
        try:
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            # Update metrics
            self.metrics['requests'] += 1
            
            return {
                'prediction': int(prediction),
                'probability': probability.tolist(),
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            self.metrics['errors'] += 1
            raise e
    
    def health_check(self):
        """Get system health"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'status': 'healthy' if self.model is not None else 'unhealthy',
            'model_loaded': self.model is not None,
            'uptime_seconds': round(uptime, 1),
            'requests': self.metrics['requests'],
            'errors': self.metrics['errors'],
            'error_rate': round(self.metrics['errors'] / max(1, self.metrics['requests']) * 100, 2)
        }

def create_simple_docker_files():
    """Create basic Docker deployment files"""
    
    # Simple Dockerfile
    dockerfile = '''FROM python:3.9-slim

WORKDIR /app
COPY . .
RUN pip install scikit-learn joblib

EXPOSE 8000
CMD ["python", "day_17_model_serving.py"]'''
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile)
    
    # Simple requirements
    requirements = '''scikit-learn==1.3.2
joblib==1.3.2
numpy==1.24.4'''
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("✅ Docker files created")

def quick_serving_demo():
    """Complete model serving demo in 15 minutes"""
    print("🚀 QUICK MODEL SERVING DEMO (15 min)")
    print("=" * 40)
    
    # 1. Initialize serving system
    print("\n1️⃣ Model Serving Setup")
    serving = QuickModelServing()
    serving.load_model()
    
    # 2. Test predictions
    print("\n2️⃣ Testing Predictions")
    test_cases = [
        [0.1, 0.2, -0.3, 0.4, -0.5],
        [-0.2, 0.8, 0.1, -0.7, 0.3],
        [0.5, -0.1, 0.9, 0.2, -0.4]
    ]
    
    for i, features in enumerate(test_cases):
        try:
            result = serving.predict(features)
            print(f"  Test {i+1}: Prediction={result['prediction']}, "
                  f"Confidence={max(result['probability']):.3f}")
        except Exception as e:
            print(f"  Test {i+1}: Error - {e}")
    
    # 3. Health monitoring
    print("\n3️⃣ Health Check")
    health = serving.health_check()
    print(f"  Status: {health['status']}")
    print(f"  Requests: {health['requests']}")
    print(f"  Error rate: {health['error_rate']}%")
    print(f"  Uptime: {health['uptime_seconds']}s")
    
    # 4. A/B Testing simulation
    print("\n4️⃣ A/B Testing Demo")
    
    # Simulate traffic split
    for i in range(10):
        version = 'v1' if np.random.random() < 0.7 else 'v2'  # 70/30 split
        features = np.random.randn(5).tolist()
        
        try:
            result = serving.predict(features)
            success = True
        except:
            success = False
        
        if i < 3:  # Show first few
            print(f"  Request {i+1}: Model={version}, Success={success}")
    
    print("  ... A/B test simulation complete")
    
    # 5. Docker deployment prep
    print("\n5️⃣ Docker Deployment")
    create_simple_docker_files()
    
    print("\n🎯 SERVING COMPLETE!")
    print("Daily practice accomplished:")
    print("  ✅ Model training & persistence")
    print("  ✅ Prediction serving system")
    print("  ✅ Health monitoring")
    print("  ✅ A/B testing simulation")
    print("  ✅ Docker deployment files")
    
    print("\n🐳 To deploy with Docker:")
    print("  docker build -t ml-model .")
    print("  docker run -p 8000:8000 ml-model")

if __name__ == "__main__":
    quick_serving_demo()