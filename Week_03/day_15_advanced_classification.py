import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ================================
# ADVANCED CLASSIFICATION SYSTEM
# ================================

class AdvancedClassificationSystem:
    """Advanced classification algorithms comparison and evaluation system"""
    
    def __init__(self):
        self.models = {
            'SVM_Linear': SVC(kernel='linear', probability=True, random_state=42),
            'SVM_RBF': SVC(kernel='rbf', probability=True, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive_Bayes_Gaussian': GaussianNB(),
            'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.trained_models = {}
        self.results = {}
        
    def create_classification_dataset(self, n_samples=1000, imbalanced=False):
        """Create synthetic classification dataset"""
        print(f"🔄 Creating classification dataset ({n_samples} samples)...")
        
        # Generate customer churn prediction data
        np.random.seed(42)
        
        # Features
        age = np.random.randint(18, 80, n_samples)
        monthly_charges = np.random.normal(65, 20, n_samples)
        total_charges = monthly_charges * np.random.randint(1, 60, n_samples)
        contract_length = np.random.choice([1, 12, 24], n_samples, p=[0.3, 0.4, 0.3])
        support_calls = np.random.poisson(2, n_samples)
        
        # Create target with business logic
        churn_probability = (
            0.3 * (age < 30).astype(int) +
            0.2 * (monthly_charges > 80).astype(int) +
            0.3 * (contract_length == 1).astype(int) +
            0.2 * (support_calls > 3).astype(int)
        )
        
        # Add noise and create binary target
        churn = (churn_probability + np.random.normal(0, 0.1, n_samples)) > 0.5
        
        # Create imbalanced dataset if requested
        if imbalanced:
            # Keep only 20% churned customers
            churn_indices = np.where(churn == 1)[0]
            keep_churn = np.random.choice(churn_indices, size=int(len(churn_indices) * 0.3), replace=False)
            no_churn_indices = np.where(churn == 0)[0]
            all_indices = np.concatenate([keep_churn, no_churn_indices])
            
            age = age[all_indices]
            monthly_charges = monthly_charges[all_indices]
            total_charges = total_charges[all_indices]
            contract_length = contract_length[all_indices]
            support_calls = support_calls[all_indices]
            churn = churn[all_indices]
        
        # Create DataFrame
        df = pd.DataFrame({
            'age': age,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'contract_length': contract_length,
            'support_calls': support_calls,
            'churn': churn.astype(int)
        })
        
        print(f"✅ Dataset created: {df.shape}")
        print(f"📊 Class distribution: {df['churn'].value_counts().to_dict()}")
        
        return df
    
    def preprocess_data(self, df, handle_imbalance=False):
        """Preprocess data with optional imbalance handling"""
        print("🔧 Preprocessing data...")
        
        # Separate features and target
        X = df.drop('churn', axis=1)
        y = df['churn']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Handle imbalanced data if requested
        if handle_imbalance:
            print("⚖️ Applying simple oversampling for imbalanced data...")
            
            # Simple oversampling using sklearn's resample
            # Create DataFrame with reset index to avoid alignment issues
            X_df = pd.DataFrame(X_train_scaled).reset_index(drop=True)
            y_series = pd.Series(y_train).reset_index(drop=True)
            
            # Separate majority and minority classes
            majority_indices = y_series == 0
            minority_indices = y_series == 1
            
            X_majority = X_df[majority_indices].reset_index(drop=True)
            X_minority = X_df[minority_indices].reset_index(drop=True)
            y_majority = y_series[majority_indices].reset_index(drop=True)
            y_minority = y_series[minority_indices].reset_index(drop=True)
            
            # Oversample minority class to match majority
            X_minority_upsampled = resample(X_minority, 
                                          replace=True,
                                          n_samples=len(X_majority),
                                          random_state=42)
            y_minority_upsampled = pd.Series([1] * len(X_minority_upsampled))
            
            # Combine majority and upsampled minority
            X_resampled = pd.concat([X_majority, X_minority_upsampled], ignore_index=True)
            y_resampled = pd.concat([y_majority, y_minority_upsampled], ignore_index=True)
            
            # Convert back to numpy arrays
            X_train_scaled = X_resampled.values
            y_train = y_resampled.values
            
            print(f"📊 After resampling: {pd.Series(y_train).value_counts().to_dict()}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all classification models"""
        print("🎯 Training and evaluating models...")
        
        for name, model in self.models.items():
            print(f"  Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                
                self.results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                self.trained_models[name] = model
                
            except Exception as e:
                print(f"    ⚠️ Error training {name}: {str(e)}")
                continue
        
        return self.results
    
    def display_results(self):
        """Display comprehensive model comparison"""
        print("\n📊 ADVANCED CLASSIFICATION RESULTS")
        print("=" * 70)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            name: {
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'ROC-AUC': results['roc_auc'],
                'CV Mean': results['cv_mean']
            }
            for name, results in self.results.items()
        }).T
        
        print(results_df.round(3))
        
        # Find best models
        print(f"\n🏆 BEST MODELS:")
        print(f"Best Accuracy: {results_df['Accuracy'].idxmax()} ({results_df['Accuracy'].max():.3f})")
        print(f"Best F1-Score: {results_df['F1-Score'].idxmax()} ({results_df['F1-Score'].max():.3f})")
        print(f"Best ROC-AUC:  {results_df['ROC-AUC'].idxmax()} ({results_df['ROC-AUC'].max():.3f})")
    
    def algorithm_insights(self):
        """Provide insights about each algorithm"""
        print(f"\n💡 ALGORITHM INSIGHTS:")
        print("=" * 50)
        
        insights = {
            'SVM_Linear': "Good for linearly separable data, fast, interpretable",
            'SVM_RBF': "Handles non-linear patterns, powerful but slower",
            'KNN': "Simple, no training phase, sensitive to feature scaling", 
            'Naive_Bayes_Gaussian': "Fast, works well with small datasets",
            'Naive_Bayes_Multinomial': "Good for text/count data",
            'Random_Forest': "Robust, handles missing values, feature importance"
        }
        
        for name, insight in insights.items():
            if name in self.results:
                score = self.results[name]['f1_score']
                print(f"{name:25}: F1={score:.3f} | {insight}")

# ================================
# API INTEGRATION FOUNDATIONS
# ================================

class MLModelAPI:
    """Foundation class for ML model API integration"""
    
    def __init__(self, model=None):
        self.model = model
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_model_for_api(self, X_train, y_train):
        """Prepare model for API serving"""
        print("🔧 Preparing model for API serving...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        print("✅ Model ready for API serving!")
    
    def predict_single(self, input_data):
        """Make single prediction (API endpoint simulation)"""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        try:
            # Validate input format
            if not isinstance(input_data, dict):
                return {"error": "Input must be dictionary"}
            
            # Expected features
            expected_features = ['age', 'monthly_charges', 'total_charges', 'contract_length', 'support_calls']
            
            if not all(feature in input_data for feature in expected_features):
                return {"error": f"Missing features. Expected: {expected_features}"}
            
            # Prepare input
            input_array = np.array([[input_data[feature] for feature in expected_features]])
            input_scaled = self.scaler.transform(input_array)
            
            # Make prediction
            prediction = self.model.predict(input_scaled)[0]
            probability = self.model.predict_proba(input_scaled)[0]
            
            return {
                "prediction": int(prediction),
                "probability_no_churn": float(probability[0]),
                "probability_churn": float(probability[1]),
                "status": "success"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def batch_predict(self, input_list):
        """Batch prediction (API endpoint simulation)"""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        results = []
        for i, input_data in enumerate(input_list):
            result = self.predict_single(input_data)
            result['id'] = i
            results.append(result)
        
        return {"predictions": results, "count": len(results)}

# ================================
# MAIN EXECUTION & TESTING
# ================================

def run_advanced_classification_system():
    """Run complete advanced classification system"""
    print("🚀 ADVANCED CLASSIFICATION + API INTEGRATION")
    print("=" * 60)
    
    # Initialize system
    classifier_system = AdvancedClassificationSystem()
    
    # Test with balanced data
    print("\n📊 BALANCED DATASET EXPERIMENT")
    df_balanced = classifier_system.create_classification_dataset(n_samples=800, imbalanced=False)
    X_train, X_test, y_train, y_test = classifier_system.preprocess_data(df_balanced, handle_imbalance=False)
    results_balanced = classifier_system.train_and_evaluate_models(X_train, X_test, y_train, y_test)
    classifier_system.display_results()
    classifier_system.algorithm_insights()
    
    # Test with imbalanced data
    print("\n📊 IMBALANCED DATASET EXPERIMENT")
    classifier_system_imb = AdvancedClassificationSystem()
    df_imbalanced = classifier_system_imb.create_classification_dataset(n_samples=800, imbalanced=True)
    X_train_imb, X_test_imb, y_train_imb, y_test_imb = classifier_system_imb.preprocess_data(df_imbalanced, handle_imbalance=True)
    results_imbalanced = classifier_system_imb.train_and_evaluate_models(X_train_imb, X_test_imb, y_train_imb, y_test_imb)
    classifier_system_imb.display_results()
    
    # API Integration Demo
    print("\n🌐 API INTEGRATION DEMO")
    print("=" * 40)
    
    # Use best model for API
    best_model_name = max(results_balanced.keys(), key=lambda x: results_balanced[x]['f1_score'])
    best_model = results_balanced[best_model_name]['model']
    
    api_service = MLModelAPI(best_model)
    api_service.prepare_model_for_api(X_train, y_train)
    
    # Test single prediction
    test_customer = {
        'age': 45,
        'monthly_charges': 85.0,
        'total_charges': 2550.0,
        'contract_length': 1,
        'support_calls': 4
    }
    
    print(f"🔮 Single Prediction Test:")
    single_result = api_service.predict_single(test_customer)
    print(f"Input: {test_customer}")
    print(f"Result: {single_result}")
    
    # Test batch prediction
    batch_customers = [
        {'age': 25, 'monthly_charges': 50.0, 'total_charges': 600.0, 'contract_length': 24, 'support_calls': 1},
        {'age': 60, 'monthly_charges': 95.0, 'total_charges': 5700.0, 'contract_length': 1, 'support_calls': 5}
    ]
    
    print(f"\n🔮 Batch Prediction Test:")
    batch_result = api_service.batch_predict(batch_customers)
    for pred in batch_result['predictions']:
        print(f"Customer {pred['id']}: Churn={pred['prediction']}, Probability={pred['probability_churn']:.3f}")
    
    print(f"\n✅ ADVANCED CLASSIFICATION COMPLETE!")
    print(f"🎯 Best performing model: {best_model_name}")
    print(f"📊 F1-Score: {results_balanced[best_model_name]['f1_score']:.3f}")
    print(f"🌐 API integration ready for FastAPI implementation!")

if __name__ == "__main__":
    run_advanced_classification_system()
    
    print(f"\n💡 DAILY PRACTICE TIPS:")
    print("• Experiment with different kernels for SVM")  
    print("• Try different k values for KNN")
    print("• Test various class weight strategies")
    print("• Practice API input validation patterns")
    print("• Build FastAPI wrapper next (Day 15 mini project)!")