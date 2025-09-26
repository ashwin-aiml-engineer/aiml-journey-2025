"""
Day 19: CI/CD for ML (15-Minute Daily Practice)  
🎯 Core ML automation: testing, deployment, monitoring
✅ Essential CI/CD patterns for ML systems
"""

import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class MLTestSuite:
    """Essential ML testing"""
    
    def test_data(self, data):
        """Test data quality"""
        tests = {
            "not_empty": len(data) > 0,
            "no_nulls": data.isnull().sum().sum() == 0,
            "valid_shape": data.shape[0] > 10 and data.shape[1] > 0
        }
        
        print("🔍 DATA TESTS")
        all_passed = True
        for test, passed in tests.items():
            print(f"   {test}: {'✅' if passed else '❌'}")
            if not passed:
                all_passed = False
        
        return all_passed
    
    def test_model(self, model, X_test, y_test):
        """Test model performance"""
        try:
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            tests = {
                "makes_predictions": len(predictions) > 0,
                "correct_output_shape": len(predictions) == len(y_test),
                "decent_accuracy": accuracy > 0.6
            }
            
            print("🎯 MODEL TESTS")
            all_passed = True
            for test, passed in tests.items():
                print(f"   {test}: {'✅' if passed else '❌'}")
                if not passed:
                    all_passed = False
            
            print(f"   accuracy: {accuracy:.4f}")
            return all_passed
            
        except Exception as e:
            print(f"❌ Model test failed: {e}")
            return False

class MLPipeline:
    """Automated ML pipeline"""
    
    def __init__(self):
        self.test_suite = MLTestSuite()
    
    def validate_and_train(self, data, target_col):
        """Complete ML pipeline with validation"""
        print("🤖 ML PIPELINE")
        print("=" * 14)
        
        # 1. Data validation
        print("\n1️⃣ DATA VALIDATION")
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        if not self.test_suite.test_data(data):
            print("❌ Data validation failed - pipeline stopped")
            return None
        
        # 2. Train model
        print("\n2️⃣ MODEL TRAINING")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        model = RandomForestClassifier(n_estimators=30, random_state=42)
        model.fit(X_train, y_train)
        print("   ✅ Model trained")
        
        # 3. Model validation
        print("\n3️⃣ MODEL VALIDATION")
        if not self.test_suite.test_model(model, X_test, y_test):
            print("❌ Model validation failed - deployment blocked")
            return None
        
        # 4. Deployment simulation
        print("\n4️⃣ DEPLOYMENT")
        deployment_steps = [
            "Building container",
            "Running security scan",
            "Deploying to staging",
            "Health check passed",
            "Promoting to production"
        ]
        
        for step in deployment_steps:
            print(f"   • {step} ✅")
        
        print("🚀 Model deployed successfully!")
        return model

class GitOpsWorkflow:
    """Simple GitOps for ML"""
    
    def simulate_workflow(self):
        """Simulate Git workflow"""
        print("📋 GITOPS WORKFLOW")
        print("=" * 17)
        
        workflow = [
            ("feature/new-model", "🔧 Develop new model"),
            ("develop", "🧪 Integration testing"),
            ("main", "🚀 Production deployment")
        ]
        
        for branch, description in workflow:
            print(f"\n📁 {branch}")
            print(f"   {description}")
            
            if "feature" in branch:
                print("   • Run unit tests ✅")
                print("   • Code review ✅") 
                print("   • Create PR ✅")
            elif branch == "develop":
                print("   • Integration tests ✅")
                print("   • Deploy to staging ✅")
            else:  # main
                print("   • Deploy to production ✅")
                print("   • Monitor metrics ✅")

class ABTesting:
    """Simple A/B testing"""
    
    def run_test(self, model_a, model_b, X_test, y_test):
        """Run A/B test between models"""
        print("\n🧪 A/B TESTING")
        print("-" * 13)
        
        # Split test data
        mid = len(X_test) // 2
        X_a, X_b = X_test[:mid], X_test[mid:]
        y_a, y_b = y_test[:mid], y_test[mid:]
        
        # Test both models
        acc_a = accuracy_score(y_a, model_a.predict(X_a))
        acc_b = accuracy_score(y_b, model_b.predict(X_b))
        
        print(f"   Model A: {acc_a:.4f}")
        print(f"   Model B: {acc_b:.4f}")
        print(f"   Winner: {'A' if acc_a > acc_b else 'B'}")
        
        return acc_a, acc_b

def cicd_demo():
    """Complete CI/CD demo in 15 minutes"""
    print("🎯 CI/CD FOR ML (15 min)")
    print("=" * 24)
    
    # Create sample data
    print("\n📊 SAMPLE DATA")
    np.random.seed(42)
    X = np.random.randn(600, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])
    data['target'] = y
    print(f"   Dataset: {len(data)} rows, {len(data.columns)} columns")
    
    # Run ML pipeline
    pipeline = MLPipeline()
    model = pipeline.validate_and_train(data, 'target')
    
    if model:
        # GitOps workflow
        print("\n📋 GITOPS")
        gitops = GitOpsWorkflow()
        gitops.simulate_workflow()
        
        # A/B testing
        model_b = RandomForestClassifier(n_estimators=50, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            data.drop('target', axis=1), data['target'], test_size=0.3
        )
        model_b.fit(X_train, y_train)
        
        ab_test = ABTesting()
        ab_test.run_test(model, model_b, X_test, y_test)
    
    print("\n🎯 CI/CD MASTERED!")
    print("✅ Automated testing")
    print("✅ Pipeline validation") 
    print("✅ GitOps workflows")
    print("✅ A/B testing")
    print("✅ Deployment automation")

if __name__ == "__main__":
    cicd_demo()