"""
Day 17: Hyperparameter Optimization (15-Minute Daily Practice)
🎯 Master Grid Search, Random Search & smart tuning quickly
✅ Essential optimization for production ML
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def quick_optimization_demo():
    """Complete hyperparameter optimization in 15 minutes"""
    print("🚀 QUICK HYPERPARAMETER OPTIMIZATION (15 min)")
    print("=" * 45)
    
    # 1. Create data
    print("🔄 Creating data...")
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X = StandardScaler().fit_transform(X)
    
    # 2. Baseline model
    print("\n📊 Baseline Performance")
    baseline = RandomForestClassifier(random_state=42)
    baseline_score = cross_val_score(baseline, X, y, cv=3, scoring='roc_auc').mean()
    print(f"  Default parameters: {baseline_score:.3f}")
    
    # 3. Grid Search (focused)
    print("\n🔍 Grid Search Optimization")
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42), 
        param_grid, cv=3, scoring='roc_auc', n_jobs=-1
    )
    grid_search.fit(X, y)
    
    print(f"  Best score: {grid_search.best_score_:.3f}")
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Improvement: {grid_search.best_score_ - baseline_score:+.3f}")
    
    # 4. Random Search (wider exploration)
    print("\n🎲 Random Search Optimization")
    param_dist = {
        'n_estimators': [30, 50, 100, 150, 200],
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_dist, n_iter=20, cv=3, scoring='roc_auc', 
        random_state=42, n_jobs=-1
    )
    random_search.fit(X, y)
    
    print(f"  Best score: {random_search.best_score_:.3f}")
    print(f"  Best params: {random_search.best_params_}")
    print(f"  Improvement: {random_search.best_score_ - baseline_score:+.3f}")
    
    # 5. Compare methods
    print("\n📈 Method Comparison")
    methods = [
        ("Baseline", baseline_score),
        ("Grid Search", grid_search.best_score_),
        ("Random Search", random_search.best_score_)
    ]
    
    best_method = max(methods, key=lambda x: x[1])
    
    for name, score in methods:
        marker = "🏆" if (name, score) == best_method else "  "
        print(f"  {marker} {name}: {score:.3f}")
    
    # 6. Quick parameter importance analysis
    print("\n🔍 Parameter Impact Analysis")
    
    # Test individual parameters
    test_params = [
        ('n_estimators', [50, 100, 200]),
        ('max_depth', [5, 10, None])
    ]
    
    for param_name, param_values in test_params:
        print(f"  {param_name} impact:")
        for value in param_values:
            params = {param_name: value}
            model = RandomForestClassifier(**params, random_state=42)
            score = cross_val_score(model, X, y, cv=3, scoring='roc_auc').mean()
            impact = score - baseline_score
            print(f"    {value}: {score:.3f} ({impact:+.3f})")
    
    print("\n🎯 OPTIMIZATION COMPLETE!")
    print("Daily practice accomplished:")
    print("  ✅ Baseline model evaluation")
    print("  ✅ Grid Search systematic tuning")
    print("  ✅ Random Search exploration")
    print("  ✅ Parameter importance analysis")
    print("  ✅ Method comparison & selection")

if __name__ == "__main__":
    quick_optimization_demo()