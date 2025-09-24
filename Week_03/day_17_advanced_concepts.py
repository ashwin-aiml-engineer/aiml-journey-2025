"""
Day 17: BONUS - Advanced ML Concepts (Concise Practice)
🚀 All advanced concepts in 20-25 minutes
🎯 SHAP, Bayesian optimization, PR curves, calibration & more
✅ Complete coverage, streamlined execution
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import precision_recall_curve, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

# Check optional imports
OPTUNA_AVAILABLE = False
SHAP_AVAILABLE = False
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    pass

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    pass

def quick_advanced_concepts():
    """All advanced concepts in streamlined format"""
    print("� ADVANCED ML CONCEPTS (20-25 min)")
    print("=" * 40)
    
    # Setup data once
    X, y = make_classification(n_samples=400, n_features=8, weights=[0.7, 0.3], random_state=42)
    X = StandardScaler().fit_transform(X)
    
    # 1. Advanced Cross-Validation (Nested CV)
    print("\n1️⃣ Nested Cross-Validation")
    from sklearn.model_selection import GridSearchCV
    
    outer_scores = []
    param_grid = {'n_estimators': [30, 50, 100]}
    
    for fold in range(3):  # Quick 3-fold
        # Simulate train/test split
        split_point = len(X) // 3
        start_idx = fold * split_point
        end_idx = (fold + 1) * split_point if fold < 2 else len(X)
        
        test_mask = np.zeros(len(X), dtype=bool)
        test_mask[start_idx:end_idx] = True
        
        X_train, X_test = X[~test_mask], X[test_mask]
        y_train, y_test = y[~test_mask], y[test_mask]
        
        # Inner CV for hyperparameter tuning
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42), 
            param_grid, cv=3, scoring='roc_auc'
        )
        grid_search.fit(X_train, y_train)
        
        # Outer evaluation
        score = grid_search.score(X_test, y_test)
        outer_scores.append(score)
    
    print(f"  Nested CV Score: {np.mean(outer_scores):.3f} (±{np.std(outer_scores):.3f})")
    
    # 2. Precision-Recall Analysis
    print("\n2️⃣ Precision-Recall Analysis")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    print(f"  PR-AUC: {pr_auc:.3f} (baseline: {np.mean(y_test):.3f})")
    
    # Best F1 threshold
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    print(f"  Best F1 threshold: {thresholds[best_f1_idx]:.3f}")
    
    # 3. SHAP Interpretability (simplified)
    print("\n3️⃣ Model Interpretability")
    if SHAP_AVAILABLE:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:20])  # Small sample
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
        
        feature_importance = np.abs(shap_values).mean(0)
        top_feature = np.argmax(feature_importance)
        print(f"  SHAP top feature: Feature_{top_feature} (importance: {feature_importance[top_feature]:.3f})")
    else:
        # Fallback to basic feature importance
        importances = model.feature_importances_
        top_feature = np.argmax(importances)
        print(f"  Top feature: Feature_{top_feature} (importance: {importances[top_feature]:.3f})")
    
    # 4. Model Calibration
    print("\n4️⃣ Model Calibration")
    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
    calibrated_model.fit(X_train, y_train)
    
    prob_uncal = model.predict_proba(X_test)[:, 1]
    prob_cal = calibrated_model.predict_proba(X_test)[:, 1]
    
    brier_uncal = brier_score_loss(y_test, prob_uncal)
    brier_cal = brier_score_loss(y_test, prob_cal)
    
    print(f"  Brier (uncalibrated): {brier_uncal:.4f}")
    print(f"  Brier (calibrated): {brier_cal:.4f}")
    print(f"  Improvement: {brier_uncal - brier_cal:+.4f}")
    
    # 5. Bayesian Optimization (simplified)
    print("\n5️⃣ Bayesian Optimization")
    if OPTUNA_AVAILABLE:
        def objective(trial):
            n_est = trial.suggest_int('n_estimators', 20, 80)
            max_d = trial.suggest_int('max_depth', 3, 12)
            
            model_opt = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42)
            return cross_val_score(model_opt, X_train, y_train, cv=3, scoring='roc_auc').mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=15, show_progress_bar=False)  # Quick trials
        
        print(f"  Best score: {study.best_value:.3f}")
        print(f"  Best params: {study.best_params}")
    else:
        # Fallback to simple random search
        best_score = 0
        best_params = {}
        
        for _ in range(10):  # Quick random trials
            n_est = np.random.choice([30, 50, 70, 100])
            max_d = np.random.choice([5, 8, 10, None])
            
            model_opt = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42)
            score = cross_val_score(model_opt, X_train, y_train, cv=3, scoring='roc_auc').mean()
            
            if score > best_score:
                best_score = score
                best_params = {'n_estimators': n_est, 'max_depth': max_d}
        
        print(f"  Best score (random): {best_score:.3f}")
        print(f"  Best params: {best_params}")
    
    # 6. Production Serving Concepts
    print("\n6️⃣ Production Serving")
    print("  ✅ Model persistence: joblib.dump/load")
    print("  ✅ Health checks: uptime, error_rate monitoring")
    print("  ✅ A/B testing: traffic splitting (70/30)")
    print("  ✅ Docker: FROM python:3.9-slim + pip install")
    print("  ✅ API: POST /predict, GET /health, GET /metrics")
    
    print("\n🎯 ADVANCED CONCEPTS COMPLETE!")
    print("Mastered in 20-25 minutes:")
    print("  ✅ Nested cross-validation for unbiased evaluation")
    print("  ✅ Precision-Recall analysis for imbalanced data")
    print("  ✅ Model interpretability (SHAP/feature importance)")
    print("  ✅ Probability calibration for reliable predictions")
    print("  ✅ Bayesian/intelligent hyperparameter optimization")
    print("  ✅ Production serving patterns & best practices")

# ================================
# COMPREHENSIVE DEMO
# ================================

def run_advanced_concepts_demo():
    """Complete advanced ML concepts demonstration"""
    print("🚀 ADVANCED ML CONCEPTS BONUS SESSION")
    print("=" * 45)
    print("🎯 All advanced topics streamlined for 20-25 min practice")
    print("=" * 45)
    
    # Run all advanced concepts in one function
    quick_advanced_concepts()

if __name__ == "__main__":
    run_advanced_concepts_demo()