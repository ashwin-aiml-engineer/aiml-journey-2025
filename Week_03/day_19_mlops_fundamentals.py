"""
Day 19: MLOps Fundamentals (15-Minute Daily Practice)
🎯 Core MLOps: experiment tracking, model registry, deployment
✅ Essential patterns for production ML
"""

import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class ExperimentTracker:
    """Simple ML experiment tracking"""
    
    def __init__(self):
        self.runs = []
        self.current_run = None
    
    def start_run(self, name):
        self.current_run = {"name": name, "params": {}, "metrics": {}}
        print(f"🚀 {name}")
    
    def log_param(self, key, value):
        if self.current_run:
            self.current_run["params"][key] = value
            print(f"📊 {key}: {value}")
    
    def log_metric(self, key, value):
        if self.current_run:
            self.current_run["metrics"][key] = value
            print(f"📈 {key}: {value:.4f}")
    
    def end_run(self):
        if self.current_run:
            self.runs.append(self.current_run)
            self.current_run = None
    
    def best_run(self, metric="accuracy"):
        return max(self.runs, key=lambda x: x["metrics"].get(metric, 0))

class ModelRegistry:
    """Simple model versioning"""
    
    def __init__(self):
        self.models_dir = Path("./models")
        self.models_dir.mkdir(exist_ok=True)
        self.registry = {}
    
    def register(self, name, model, stage="dev"):
        version = len(self.registry.get(name, [])) + 1
        model_path = self.models_dir / f"{name}_v{version}.joblib"
        
        # Save model
        joblib.dump(model, model_path)
        
        # Track in registry
        if name not in self.registry:
            self.registry[name] = []
        
        self.registry[name].append({
            "version": version,
            "stage": stage,
            "path": str(model_path),
            "created": datetime.now().isoformat()
        })
        
        print(f"✅ Registered {name} v{version} ({stage})")
        return version
    
    def promote(self, name, version, stage):
        if name in self.registry and version <= len(self.registry[name]):
            self.registry[name][version-1]["stage"] = stage
            print(f"🚀 Promoted {name} v{version} to {stage}")
    
    def load(self, name, stage="prod"):
        if name in self.registry:
            for model_info in reversed(self.registry[name]):
                if model_info["stage"] == stage:
                    return joblib.load(model_info["path"])
        return None

def mlops_demo():
    """Complete MLOps workflow in 15 minutes"""
    print("🎯 MLOPS ESSENTIALS (15 min)")
    print("=" * 30)
    
    # 1. Data
    print("\n1️⃣ DATA")
    np.random.seed(42)
    X = np.random.randn(800, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print(f"   Training: {X_train.shape[0]} samples")
    
    # 2. Experiment Tracking
    print("\n2️⃣ EXPERIMENTS")
    tracker = ExperimentTracker()
    
    models = {
        "rf_small": RandomForestClassifier(n_estimators=20, random_state=42),
        "rf_large": RandomForestClassifier(n_estimators=50, random_state=42)
    }
    
    for name, model in models.items():
        tracker.start_run(name)
        tracker.log_param("n_estimators", model.n_estimators)
        
        model.fit(X_train, y_train)
        score = accuracy_score(y_test, model.predict(X_test))
        tracker.log_metric("accuracy", score)
        tracker.end_run()
    
    # 3. Model Registry
    print("\n3️⃣ MODEL REGISTRY")
    registry = ModelRegistry()
    
    best = tracker.best_run()
    best_model = models[best["name"]]
    
    # Register and promote
    version = registry.register("classifier", best_model)
    registry.promote("classifier", version, "staging")
    registry.promote("classifier", version, "prod")
    
    # 4. Deployment Simulation
    print("\n4️⃣ DEPLOYMENT")
    prod_model = registry.load("classifier", "prod")
    
    if prod_model:
        # Test production model
        test_pred = prod_model.predict(X_test[:5])
        print(f"   Production predictions: {test_pred}")
        print("   ✅ Model deployed and working")
    
    print(f"\n🏆 Best model: {best['name']} ({best['metrics']['accuracy']:.4f})")
    print("\n🎯 MLOPS MASTERED!")
    print("✅ Experiment tracking")
    print("✅ Model versioning") 
    print("✅ Stage promotion")
    print("✅ Production deployment")

if __name__ == "__main__":
    mlops_demo()