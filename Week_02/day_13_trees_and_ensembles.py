# Topic 1: Decision Tree Fundamentals
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Generate sample data
X = np.random.rand(100, 4)
y = np.random.randint(0, 2, 100)

# Basic Decision Tree
tree = DecisionTreeClassifier(criterion='gini', max_depth=3)
tree.fit(X, y)
pred = tree.predict(X)
print("Tree Accuracy:", accuracy_score(y, pred))
print("Feature Importance:", tree.feature_importances_)

# Topic 2: Decision Tree Construction & Pruning
tree_pruned = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=3,
    ccp_alpha=0.01  # Cost complexity pruning
)
tree_pruned.fit(X, y)
print("Pruned Tree Accuracy:", accuracy_score(y, tree_pruned.predict(X)))

# Topic 3: Decision Tree Analysis
print("Tree Depth:", tree.get_depth())
print("Number of Leaves:", tree.get_n_leaves())

# Topic 4: Random Forest Fundamentals
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True
)
rf.fit(X, y)
print("RF Accuracy:", accuracy_score(y, rf.predict(X)))
print("OOB Score:", rf.oob_score_)
print("RF Feature Importance:", rf.feature_importances_)

# Topic 5: Advanced Random Forest Techniques
rf_advanced = RandomForestClassifier(
    n_estimators=200,
    max_features=2,
    class_weight='balanced'
)
rf_advanced.fit(X, y)
print("Advanced RF Accuracy:", accuracy_score(y, rf_advanced.predict(X)))

# Topic 6: Gradient Boosting Fundamentals
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)
gb.fit(X, y)
print("GB Accuracy:", accuracy_score(y, gb.predict(X)))
print("GB Feature Importance:", gb.feature_importances_)

# Topic 7: Advanced Boosting Algorithms
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=1.0
)
ada.fit(X, y)
print("AdaBoost Accuracy:", accuracy_score(y, ada.predict(X)))

# Topic 8: Ensemble Method Comparison
from sklearn.ensemble import VotingClassifier

# Voting Classifier
voting = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('ada', ada)],
    voting='soft'
)
voting.fit(X, y)
print("Voting Accuracy:", accuracy_score(y, voting.predict(X)))

# Test all components
if __name__ == "__main__":
    print("\n--- Decision Tree Test ---")
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)
    tree = DecisionTreeClassifier(criterion='gini', max_depth=3).fit(X, y)
    print("Tree Accuracy:", accuracy_score(y, tree.predict(X)))
    print("Feature Importance:", tree.feature_importances_)
    print("Tree Depth:", tree.get_depth())
    
    print("\n--- Random Forest Test ---")
    rf = RandomForestClassifier(n_estimators=100, oob_score=True).fit(X, y)
    print("RF Accuracy:", accuracy_score(y, rf.predict(X)))
    print("OOB Score:", rf.oob_score_)
    print("RF Feature Importance:", rf.feature_importances_)
    
    print("\n--- Gradient Boosting Test ---")
    gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1).fit(X, y)
    print("GB Accuracy:", accuracy_score(y, gb.predict(X)))
    print("GB Feature Importance:", gb.feature_importances_)
    
    print("\n--- AdaBoost Test ---")
    ada = AdaBoostClassifier(n_estimators=50).fit(X, y)
    print("AdaBoost Accuracy:", accuracy_score(y, ada.predict(X)))
    
    print("\n--- Voting Classifier Test ---")
    voting = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('ada', ada)],
        voting='soft'
    ).fit(X, y)
    print("Voting Accuracy:", accuracy_score(y, voting.predict(X)))