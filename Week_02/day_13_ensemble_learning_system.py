# Project Topic 1: Intelligent Decision Tree Builder
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectFromModel

def intelligent_tree_builder(X, y):
    """Automated tree construction with optimal parameters"""
    param_grid = {
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5]
    }
    
    tree = DecisionTreeClassifier()
    grid_search = GridSearchCV(tree, param_grid, cv=3)
    grid_search.fit(X, y)
    
    print("Best Tree Params:", grid_search.best_params_)
    print("Best Tree Score:", grid_search.best_score_)
    return grid_search.best_estimator_

# Project Topic 2: Advanced Random Forest Engine
def advanced_rf_engine(X, y):
    """Random Forest with hyperparameter optimization"""
    rf = RandomForestClassifier(
        n_estimators=100,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True
    )
    rf.fit(X, y)
    
    print("RF OOB Score:", rf.oob_score_)
    print("RF Feature Importance:", rf.feature_importances_)
    return rf

# Project Topic 3: Gradient Boosting Optimizer
def gb_optimizer(X, y):
    """Gradient Boosting with learning rate tuning"""
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    gb.fit(X, y)
    
    print("GB Train Score:", gb.train_score_[-1])
    # validation_score_ only available with validation_fraction and early stopping
    if hasattr(gb, 'validation_score_') and len(gb.validation_score_) > 0:
        print("GB Validation Score:", gb.validation_score_[-1])
    else:
        print("GB Validation Score: Not available (requires early stopping)")
    return gb

# Project Topic 4: Ensemble Method Comparator
def ensemble_comparator(X, y):
    """Compare different ensemble methods"""
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=50),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=50),
        'AdaBoost': AdaBoostClassifier(n_estimators=50)
    }
    
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=3)
        results[name] = np.mean(scores)
        print(f"{name} CV Score: {np.mean(scores):.3f}")
    
    return results

# Project Topic 5: Tree-Based Feature Selection System
def tree_feature_selection(X, y):
    """Feature selection using tree-based importance"""
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X, y)
    
    selector = SelectFromModel(rf, prefit=True)
    X_selected = selector.transform(X)
    
    print("Original Features:", X.shape[1])
    print("Selected Features:", X_selected.shape[1])
    print("Feature Importance:", rf.feature_importances_)
    return X_selected, rf.feature_importances_

# Project Topic 6: Production Ensemble Pipeline
def production_ensemble_pipeline(X, y):
    """Multi-algorithm ensemble system"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Individual models
    rf = RandomForestClassifier(n_estimators=50)
    gb = GradientBoostingClassifier(n_estimators=50)
    ada = AdaBoostClassifier(n_estimators=50)
    
    # Voting ensemble
    voting = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('ada', ada)],
        voting='soft'
    )
    
    voting.fit(X_train, y_train)
    pred = voting.predict(X_test)
    
    print("Ensemble Test Accuracy:", accuracy_score(y_test, pred))
    
    # Model persistence simulation
    import joblib
    joblib.dump(voting, 'ensemble_model.joblib')
    print("Ensemble model saved as ensemble_model.joblib")
    
    return voting

# Test all components
if __name__ == "__main__":
    # Generate sample data
    X = np.random.rand(200, 5)
    y = np.random.randint(0, 2, 200)
    
    print("--- Intelligent Decision Tree Builder Test ---")
    best_tree = intelligent_tree_builder(X, y)
    
    print("\n--- Advanced Random Forest Engine Test ---")
    rf_model = advanced_rf_engine(X, y)
    
    print("\n--- Gradient Boosting Optimizer Test ---")
    gb_model = gb_optimizer(X, y)
    
    print("\n--- Ensemble Method Comparator Test ---")
    comparison = ensemble_comparator(X, y)
    
    print("\n--- Tree-Based Feature Selection System Test ---")
    X_selected, importances = tree_feature_selection(X, y)
    
    print("\n--- Production Ensemble Pipeline Test ---")
    ensemble_model = production_ensemble_pipeline(X, y)