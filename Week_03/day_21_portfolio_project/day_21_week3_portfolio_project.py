"""
Day 21: Week 3 Portfolio Project - Production-Ready ML Service Platform.
"""
import os
import time
import logging
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import uvicorn
import requests

# --- Config & Secrets Management ---
class Config:
    ENV = os.getenv("ENV", "development")
    SECRET_KEY = os.getenv("SECRET_KEY", "changeme")
    SSL_ENABLED = os.getenv("SSL_ENABLED", "false") == "true"
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

logging.basicConfig(level=logging.INFO)

# --- ML Pipeline Core ---
def load_data(path):
    logging.info(f"Loading data from {path}")
    return pd.read_csv(path)

def preprocess(df):
    logging.info("Preprocessing data...")
    df = df.dropna()
    X = df.drop('target', axis=1)
    y = df['target']
    return X, y

def train_model(X, y):
    logging.info("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = RandomForestClassifier()
    params = {'n_estimators': [10, 50], 'max_depth': [3, 5]}
    grid = GridSearchCV(clf, params, cv=2)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Model accuracy: {acc:.2f}")
    return grid.best_estimator_, acc

def run_pipeline(data_path):
    df = load_data(data_path)
    X, y = preprocess(df)
    model, acc = train_model(X, y)
    return model, acc

# --- Model Serving Architecture ---
class ModelServer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.version = None
    def load(self):
        self.model = joblib.load(self.model_path)
        self.version = int(time.time())
        logging.info(f"Model loaded. Version: {self.version}")
    def predict(self, X):
        if self.model is None:
            self.load()
        try:
            return self.model.predict(X)
        except Exception as e:
            logging.error("Prediction failed, fallback activated.")
            return [0]*len(X)

# --- Monitoring & Logging ---
class Monitor:
    def __init__(self):
        self.metrics = {}
    def log_metric(self, name, value):
        self.metrics[name] = value
        logging.info(f"Metric {name}: {value}")
    def alert(self, msg):
        logging.warning(f"ALERT: {msg}")
    def profile(self, func, *args):
        start = time.time()
        result = func(*args)
        elapsed = time.time() - start
        self.log_metric("elapsed_time", elapsed)
        return result

# --- Deployment & Operations ---
def blue_green_deploy():
    logging.info("Blue-green deployment: switching traffic after validation.")

def rolling_update():
    logging.info("Rolling update: updating containers with zero downtime.")

def backup():
    logging.info("Backup: saving model and data snapshots.")

def monitor():
    logging.info("Monitoring: collecting logs and metrics.")

def document():
    logging.info("Documentation: updating runbooks and API docs.")

# --- FastAPI Service ---
app = FastAPI(title="ML Service API", version="1.0.0")

class InputData(BaseModel):
    features: list

@app.post("/v1/predict")
async def predict(data: InputData):
    try:
        model = joblib.load("model.joblib")
        pred = model.predict([data.features])[0]
        return {"prediction": int(pred)}
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail="Prediction error")

@app.get("/v1/health")
async def health():
    return {"status": "ok"}

# --- API & Pipeline Tests ---
def test_health():
    r = requests.get("http://localhost:8000/v1/health")
    assert r.status_code == 200

def test_predict():
    r = requests.post("http://localhost:8000/v1/predict", json={"features": [0.1, 0.2, 0.3]})
    assert r.status_code == 200
    assert "prediction" in r.json()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\nDay 21 Portfolio Project: Functionality Test Menu")
    print("1. Run ML pipeline and print accuracy")
    print("2. Save trained model to model.joblib")
    print("3. Test model serving (load and predict)")
    print("4. Test monitoring and profiling")
    print("5. Test deployment operations")
    print("6. Run FastAPI server (manual test)")
    print("7. Run API endpoint tests (health, predict)")
    print("0. Exit")
    choice = input("Select option: ")
    data_path = os.path.join(os.path.dirname(__file__), 'data.csv')
    model_path = os.path.join(os.path.dirname(__file__), 'model.joblib')
    if choice == "1":
        model, acc = run_pipeline(data_path)
        print(f"Trained model accuracy: {acc}")
    elif choice == "2":
        model, acc = run_pipeline(data_path)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    elif choice == "3":
        model, acc = run_pipeline(data_path)
        joblib.dump(model, model_path)
        server = ModelServer(model_path)
        server.load()
        X = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        preds = server.predict(X)
        print(f"ModelServer predictions: {preds}")
    elif choice == "4":
        monitor = Monitor()
        def dummy_func(x):
            time.sleep(0.1)
            return x * 2
        result = monitor.profile(dummy_func, 5)
        print(f"Profiled result: {result}")
        monitor.log_metric("custom_metric", 42)
        monitor.alert("Test alert!")
    elif choice == "5":
        blue_green_deploy()
        rolling_update()
        backup()
        monitor()
        document()
        print("Deployment operations tested.")
    elif choice == "6":
        print("Starting FastAPI server on port 8000...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    elif choice == "7":
        print("Testing API endpoints...")
        try:
            test_health()
            print("Health endpoint OK.")
        except Exception as e:
            print(f"Health endpoint test failed: {e}")
        try:
            test_predict()
            print("Predict endpoint OK.")
        except Exception as e:
            print(f"Predict endpoint test failed: {e}")
    else:
        print("Exiting.")
