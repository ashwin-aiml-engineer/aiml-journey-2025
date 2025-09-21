import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ================================
# ADVANCED PYTHON (Day 8 Learning)
# ================================

def timing_decorator(func):
    """Decorator to time function execution"""
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"⏱️ {func.__name__} took {end-start:.3f} seconds")
        return result
    return wrapper

def data_generator(n_samples=1000):
    """Generator for memory-efficient data creation"""
    for i in range(n_samples):
        # Simulate employee data
        yield {
            'age': np.random.randint(22, 65),
            'experience': np.random.randint(0, 40),
            'salary': np.random.randint(30000, 150000),
            'department': np.random.choice(['Engineering', 'Sales', 'Marketing']),
            'performance': np.random.choice(['Good', 'Average', 'Excellent'])
        }

# ================================
# ML PIPELINE CLASS (OOP from Day 8)
# ================================

class SimpleMLPipeline:
    """Simple ML pipeline demonstrating Week 2 concepts"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {
            'logistic': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.trained_models = {}
        
    @timing_decorator
    def create_dataset(self, n_samples=1000):
        """Create synthetic dataset using generator (Day 8 concept)"""
        print(f"🔄 Creating dataset with {n_samples} samples...")
        
        # Use generator for memory efficiency
        data = list(data_generator(n_samples))
        df = pd.DataFrame(data)
        
        print(f"✅ Dataset created: {df.shape}")
        return df
    
    @timing_decorator  
    def preprocess_data(self, df):
        """Data preprocessing (Day 11 learning)"""
        print("🔧 Preprocessing data...")
        
        # Handle categorical variables
        dept_encoded = pd.get_dummies(df['department'], prefix='dept')
        
        # Combine features
        X = pd.concat([
            df[['age', 'experience', 'salary']], 
            dept_encoded
        ], axis=1)
        
        # Encode target
        y = self.label_encoder.fit_transform(df['performance'])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        print(f"✅ Features prepared: {X_scaled.shape}")
        print(f"✅ Target classes: {self.label_encoder.classes_}")
        
        return X_scaled, y
    
    @timing_decorator
    def train_models(self, X, y):
        """Train multiple models (Day 12-13 learning)"""
        print("🎯 Training models...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        
        for name, model in self.models.items():
            print(f"  Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            results[name] = {
                'model': model,
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            self.trained_models[name] = model
            
        return results, X_test, y_test
    
    def evaluate_models(self, results, X_test, y_test):
        """Model evaluation and comparison"""
        print("\n📊 MODEL COMPARISON RESULTS")
        print("=" * 50)
        
        for name, result in results.items():
            print(f"\n🔍 {name.upper()}")
            print(f"  Train Accuracy: {result['train_accuracy']:.3f}")
            print(f"  Test Accuracy:  {result['test_accuracy']:.3f}")
            print(f"  CV Score:       {result['cv_mean']:.3f} ± {result['cv_std']:.3f}")
            
            # Detailed classification report for best model
            if name == 'random_forest':
                y_pred = result['model'].predict(X_test)
                print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
                print(classification_report(y_test, y_pred, 
                                          target_names=self.label_encoder.classes_))
    
    def feature_importance_analysis(self):
        """Feature importance (Day 13 learning)"""
        if 'random_forest' not in self.trained_models:
            return
            
        rf_model = self.trained_models['random_forest']
        feature_names = ['age', 'experience', 'salary', 'dept_Engineering', 'dept_Marketing', 'dept_Sales']
        
        importance = rf_model.feature_importances_
        
        print("\n🎯 FEATURE IMPORTANCE (Random Forest)")
        print("=" * 40)
        for name, imp in zip(feature_names, importance):
            print(f"  {name:15}: {imp:.3f}")
    
    def run_complete_pipeline(self, n_samples=1000):
        """Complete ML pipeline demonstrating all Week 2 concepts"""
        print("🚀 STARTING WEEK 2 ML PIPELINE")
        print("=" * 50)
        
        # Step 1: Create data
        df = self.create_dataset(n_samples)
        
        # Step 2: Preprocess
        X, y = self.preprocess_data(df)
        
        # Step 3: Train models
        results, X_test, y_test = self.train_models(X, y)
        
        # Step 4: Evaluate
        self.evaluate_models(results, X_test, y_test)
        
        # Step 5: Feature analysis
        self.feature_importance_analysis()
        
        print("\n✅ PIPELINE COMPLETE - All Week 2 concepts demonstrated!")
        return results

# ================================
# MAIN EXECUTION
# ================================

if __name__ == "__main__":
    # Create and run simple ML pipeline
    pipeline = SimpleMLPipeline()
    
    # Run with different dataset sizes for practice
    print("📚 WEEK 2 PORTFOLIO PROJECT - SIMPLE ML PIPELINE")
    print("Demonstrates: Advanced Python + ML Fundamentals")
    print("\n" + "="*60)
    
    # Small dataset for quick testing
    results = pipeline.run_complete_pipeline(n_samples=500)
    
    print("\n" + "="*60)
    print("🎉 WEEK 2 LEARNING DEMONSTRATED:")
    print("✅ Decorators for timing")
    print("✅ Generators for data creation")
    print("✅ OOP pipeline design") 
    print("✅ Data preprocessing")
    print("✅ Multiple ML algorithms")
    print("✅ Model evaluation")
    print("✅ Feature importance")
    print("✅ Cross-validation")