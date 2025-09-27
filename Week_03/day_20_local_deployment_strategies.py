"""
Day 20: Local Deployment Strategies (15-Minute Practice)
🎯 On-premise, edge, offline-first, optimization
"""

import time
import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import os
import threading
import random

# Simulate a simple ML model
class DummyModel:
    def predict(self, X):
        time.sleep(0.01)  # Simulate computation
        return [int(x.sum() > 0) for x in X]

# Local hardware optimization
class LocalOptimizer:
    def __init__(self):
        self.cache = {}
        self.max_cache_size = 100
    def cache_prediction(self, key, value):
        if len(self.cache) >= self.max_cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value
    def get_cached(self, key):
        return self.cache.get(key)

# FastAPI local model server
app = FastAPI()
model = DummyModel()
optimizer = LocalOptimizer()

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    key = str(data.features)
    cached = optimizer.get_cached(key)
    if cached is not None:
        return {"prediction": cached, "cached": True}
    pred = model.predict([np.array(data.features)])
    optimizer.cache_prediction(key, pred[0])
    return {"prediction": pred[0], "cached": False}

# Edge computing simulation
class EdgeDevice:
    def __init__(self, device_id):
        self.device_id = device_id
        self.model = DummyModel()
    def run_inference(self, X):
        # Simulate resource constraint
        if random.random() < 0.1:
            time.sleep(0.1)  # Simulate slow device
        return self.model.predict(X)

def offline_first_demo():
    print("\nOffline-first ML application demo:")
    # Simulate local cache
    cache = {}
    for i in range(5):
        x = np.random.randn(3)
        key = tuple(x)
        if key in cache:
            print(f"Cached prediction for {key}: {cache[key]}")
        else:
            pred = model.predict([x])[0]
            cache[key] = pred
            print(f"Computed prediction for {key}: {pred}")

# Resource-constrained deployment
class ResourceManager:
    def __init__(self, max_memory_mb=128):
        self.max_memory_mb = max_memory_mb
    def optimize(self):
        print(f"Optimizing for max memory: {self.max_memory_mb}MB")
        # Simulate quantization
        print("Model quantized for local deployment.")

# Client-side vs server-side inference
class InferenceDemo:
    def client_side(self, X):
        print("Client-side inference:")
        return model.predict(X)
    def server_side(self, X):
        print("Server-side inference:")
        return model.predict(X)

def local_deployment_demo():
    print("\nDay 20: Local Deployment Strategies")
    print("=" * 32)
    print("1️⃣ On-premise model serving (FastAPI)")
    print("2️⃣ Edge computing simulation")
    print("3️⃣ Offline-first ML application")
    print("4️⃣ Resource-constrained deployment")
    print("5️⃣ Client-side vs server-side inference")
    print("6️⃣ Local model caching and optimization")
    print("\n--- Practical Demos ---")
    # Edge device demo
    edge = EdgeDevice("edge-01")
    print("Edge device inference:", edge.run_inference([np.random.randn(3)]))
    # Offline-first demo
    offline_first_demo()
    # Resource manager demo
    resource_mgr = ResourceManager(64)
    resource_mgr.optimize()
    # Inference demo
    inf_demo = InferenceDemo()
    print("Client-side:", inf_demo.client_side([np.random.randn(3)]))
    print("Server-side:", inf_demo.server_side([np.random.randn(3)]))
    print("\nAll local deployment strategies covered!")

if __name__ == "__main__":
    local_deployment_demo()
