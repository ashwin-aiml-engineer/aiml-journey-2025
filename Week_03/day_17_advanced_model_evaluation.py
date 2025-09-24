"""
Day 17: Advanced Model Evaluation (15-Minute Daily Practice)
🎯 Master cross-validation, learning curves & interpretability quickly
✅ Essential evaluation for production ML
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def quick_evaluation_demo():
    """Complete model evaluation in 15 minutes"""
    print("🚀 QUICK MODEL EVALUATION (15 min)")
    print("=" * 35)
    
    # 1. Create data
    print("🔄 Creating data...")
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X = StandardScaler().fit_transform(X)
    
    # 2. Cross-validation
    print("\n📊 Cross-Validation Analysis")
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print(f"  ROC-AUC: {scores.mean():.3f} (±{scores.std():.3f})")
    
    # 3. Learning curves (quick version)
    print("\n📈 Learning Curves Check")
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=3, train_sizes=[0.3, 0.6, 1.0], scoring='roc_auc'
    )
    
    final_train = train_scores.mean(axis=1)[-1]
    final_val = val_scores.mean(axis=1)[-1]
    gap = final_train - final_val
    
    print(f"  Training: {final_train:.3f}, Validation: {final_val:.3f}")
    print(f"  Gap: {gap:.3f} {'⚠️ Overfitting!' if gap > 0.05 else '✅ Good'}")
    
    # 4. ROC analysis
    print("\n🎯 ROC Analysis")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Find best threshold
    optimal_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[optimal_idx]
    
    print(f"  ROC-AUC: {roc_auc:.3f}")
    print(f"  Best Threshold: {best_threshold:.3f}")
    
    # 5. Feature importance (quick interpretability)
    print("\n🔍 Top Features")
    importances = model.feature_importances_
    top_features = np.argsort(importances)[::-1][:3]
    
    for i, feat_idx in enumerate(top_features):
        print(f"  {i+1}. Feature_{feat_idx}: {importances[feat_idx]:.3f}")
    
    # 6. Quick visualization
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Train')
    plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Val')
    plt.title('Learning Curves')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('quick_model_evaluation.png', dpi=150)
    print("\n✅ Saved: quick_model_evaluation.png")
    plt.show()
    
    print("\n🎯 EVALUATION COMPLETE!")
    print("Daily practice accomplished:")
    print("  ✅ Cross-validation assessment")
    print("  ✅ Overfitting detection")
    print("  ✅ ROC analysis & threshold optimization")
    print("  ✅ Feature importance ranking")

if __name__ == "__main__":
    quick_evaluation_demo()