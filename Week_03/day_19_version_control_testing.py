"""
Day 19: Version Control & Testing (15-Minute Daily Practice)
🎯 Core Git workflows and ML testing essentials  
✅ Data versioning, collaboration, automated testing
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class GitWorkflow:
    """Essential Git workflows for ML"""
    
    def show_branching_strategy(self):
        """ML-specific Git branching"""
        print("📋 GIT WORKFLOW")
        print("=" * 15)
        
        branches = [
            ("main", "🌟 Production models"),
            ("develop", "🧪 Integration & testing"),
            ("feature/model-v2", "🔧 New model development"),
            ("hotfix/bug-fix", "🚨 Critical production fixes")
        ]
        
        for branch, description in branches:
            print(f"\n📁 {branch}")
            print(f"   {description}")
            
            if "feature/" in branch:
                print("   • Create from develop")
                print("   • Run local tests")
                print("   • Create pull request")
            elif branch == "develop":
                print("   • Merge features")
                print("   • Run CI/CD pipeline")
                print("   • Deploy to staging")
    
    def show_lfs_setup(self):
        """Git LFS for ML files"""
        print("\n💾 GIT LFS SETUP")
        print("-" * 16)
        
        lfs_files = ["*.csv", "*.pkl", "*.joblib", "*.h5", "*.parquet"]
        commands = [
            "git lfs install",
            "git lfs track '*.pkl'",
            "git add .gitattributes"
        ]
        
        print("📦 Large files to track:")
        for file_type in lfs_files:
            print(f"   • {file_type}")
        
        print("\n⚡ Setup commands:")
        for cmd in commands:
            print(f"   $ {cmd}")

class DataVersioning:
    """Simple data versioning"""
    
    def __init__(self):
        self.versions = {}
        self.data_dir = Path("./data_versions")
        self.data_dir.mkdir(exist_ok=True)
    
    def version_data(self, data, name):
        """Create data version"""
        # Create hash for data integrity
        data_hash = hashlib.md5(str(data.values).encode()).hexdigest()[:8]
        
        version_info = {
            "name": name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "shape": data.shape,
            "hash": data_hash,
            "file": f"{name}_{data_hash}.csv"
        }
        
        # Save data
        data.to_csv(self.data_dir / version_info["file"], index=False)
        
        # Track version
        if name not in self.versions:
            self.versions[name] = []
        self.versions[name].append(version_info)
        
        print(f"📦 Versioned: {name}")
        print(f"   Shape: {data.shape}")
        print(f"   Hash: {data_hash}")
        
        return data_hash
    
    def list_versions(self):
        """Show all data versions"""
        print("\n📚 DATA VERSIONS")
        print("-" * 16)
        
        for dataset, versions in self.versions.items():
            print(f"\n📊 {dataset}")
            for v in versions:
                print(f"   v{v['hash']}: {v['timestamp']} {v['shape']}")

class MLTesting:
    """Essential ML tests"""
    
    def test_data_quality(self, data):
        """Basic data quality tests"""
        print("🔍 DATA QUALITY TESTS")
        tests = [
            ("Has data", len(data) > 0),
            ("No nulls", data.isnull().sum().sum() == 0),
            ("Valid dtypes", len(data.select_dtypes(include=[np.number]).columns) > 0)
        ]
        
        all_pass = True
        for test_name, result in tests:
            print(f"   {test_name}: {'✅' if result else '❌'}")
            if not result:
                all_pass = False
        
        return all_pass
    
    def test_model_behavior(self, model, X_sample, y_sample):
        """Basic model behavior tests"""
        print("\n🤖 MODEL TESTS")
        
        try:
            predictions = model.predict(X_sample)
            accuracy = accuracy_score(y_sample, predictions)
            
            tests = [
                ("Produces output", len(predictions) > 0),
                ("Correct shape", len(predictions) == len(X_sample)),
                ("Reasonable accuracy", accuracy > 0.5)
            ]
            
            all_pass = True
            for test_name, result in tests:
                print(f"   {test_name}: {'✅' if result else '❌'}")
                if not result:
                    all_pass = False
            
            print(f"   Accuracy: {accuracy:.4f}")
            return all_pass
            
        except Exception as e:
            print(f"   ❌ Model failed: {e}")
            return False

class TeamCollaboration:
    """ML team collaboration practices"""
    
    def show_review_checklist(self):
        """Code review checklist for ML"""
        print("\n👥 CODE REVIEW CHECKLIST")
        print("-" * 25)
        
        checklist = {
            "Code Quality": [
                "Functions documented",
                "No hardcoded values",
                "Error handling included"
            ],
            "ML Specific": [
                "Random seeds set", 
                "Data validation added",
                "Model metrics logged"
            ],
            "Testing": [
                "Unit tests included",
                "Model tests pass",
                "Data pipeline tested"
            ]
        }
        
        for category, items in checklist.items():
            print(f"\n📍 {category}:")
            for item in items:
                print(f"   ☐ {item}")

def version_control_demo():
    """Complete version control demo in 15 minutes"""
    print("🎯 VERSION CONTROL & TESTING (15 min)")
    print("=" * 39)
    
    # Git workflow
    git = GitWorkflow()
    git.show_branching_strategy()
    git.show_lfs_setup()
    
    # Sample data
    print("\n📊 SAMPLE DATA")
    np.random.seed(42)
    X = np.random.randn(400, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    data = pd.DataFrame(X, columns=['feat_a', 'feat_b', 'feat_c'])
    data['target'] = y
    print(f"   Dataset: {data.shape}")
    
    # Data versioning
    versioning = DataVersioning()
    versioning.version_data(data, "initial_data")
    
    # Add new feature and version again
    data['feat_d'] = np.random.randn(len(data))
    versioning.version_data(data, "enhanced_data")
    versioning.list_versions()
    
    # ML testing
    print("\n🧪 ML TESTING")
    testing = MLTesting()
    
    # Test data
    data_quality_ok = testing.test_data_quality(data.drop('target', axis=1))
    
    if data_quality_ok:
        # Train and test model
        X_train, X_test, y_train, y_test = train_test_split(
            data.drop('target', axis=1), data['target'], test_size=0.3
        )
        
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_train, y_train)
        
        testing.test_model_behavior(model, X_test, y_test)
    
    # Team collaboration
    collaboration = TeamCollaboration()
    collaboration.show_review_checklist()
    
    print("\n🎯 VERSION CONTROL MASTERED!")
    print("✅ Git workflows & LFS")
    print("✅ Data versioning")
    print("✅ Automated ML testing") 
    print("✅ Team collaboration")

if __name__ == "__main__":
    version_control_demo()