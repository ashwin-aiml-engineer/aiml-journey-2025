"""
Day 20: Cloud Deployment Patterns (15-Minute Practice)
🎯 Multi-cloud, serverless, containers, scaling, cost
"""

import time
import random

# Cloud platform comparison
class CloudPlatform:
    def __init__(self, name):
        self.name = name
    def deploy_model(self):
        print(f"Deploying model to {self.name}...")
        time.sleep(0.1)
        print(f"Model deployed on {self.name}!")

# Serverless ML deployment
class ServerlessML:
    def deploy(self):
        print("Deploying ML model as serverless function...")
        print("Auto-scaling enabled. Cost optimized.")

# Container orchestration (Kubernetes basics)
class KubeCluster:
    def __init__(self, nodes=3):
        self.nodes = nodes
    def deploy(self):
        print(f"Deploying model on Kubernetes cluster with {self.nodes} nodes...")
        print("Auto-scaling and rolling updates enabled.")

# Cloud storage integration
class CloudStorage:
    def __init__(self, provider):
        self.provider = provider
    def store_artifact(self, artifact):
        print(f"Storing {artifact} in {self.provider} cloud storage.")

# Cost optimization strategies
class CostOptimizer:
    def optimize(self, usage):
        print(f"Optimizing cost for usage: {usage} units.")
        print("Spot instances and auto-scaling applied.")

# Multi-cloud deployment
class MultiCloud:
    def __init__(self, platforms):
        self.platforms = platforms
    def deploy_all(self):
        print("Deploying model to multiple clouds:")
        for p in self.platforms:
            p.deploy_model()
        print("Multi-cloud deployment complete.")

# Cloud deployment demo
def cloud_deployment_demo():
    print("\nDay 20: Cloud Deployment Patterns")
    print("=" * 32)
    print("1️⃣ Cloud platform comparison (AWS, Azure, GCP)")
    print("2️⃣ Serverless ML deployment concepts")
    print("3️⃣ Container orchestration (Kubernetes)")
    print("4️⃣ Auto-scaling for ML workloads")
    print("5️⃣ Cloud storage integration")
    print("6️⃣ Cost optimization strategies")
    print("7️⃣ Multi-cloud deployment considerations")
    print("\n--- Practical Demos ---")
    # Cloud platform demo
    aws = CloudPlatform("AWS")
    azure = CloudPlatform("Azure")
    gcp = CloudPlatform("GCP")
    aws.deploy_model()
    azure.deploy_model()
    gcp.deploy_model()
    # Serverless demo
    serverless = ServerlessML()
    serverless.deploy()
    # Kubernetes demo
    kube = KubeCluster(nodes=5)
    kube.deploy()
    # Cloud storage demo
    storage = CloudStorage("AWS")
    storage.store_artifact("model.pkl")
    # Cost optimizer demo
    cost_opt = CostOptimizer()
    cost_opt.optimize(500)
    # Multi-cloud demo
    multi = MultiCloud([aws, azure, gcp])
    multi.deploy_all()
    print("\nAll cloud deployment patterns covered!")

if __name__ == "__main__":
    cloud_deployment_demo()
