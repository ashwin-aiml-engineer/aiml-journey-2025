# Day 10 Mini Project: ML Algorithm Comparison & Advanced Evaluation
# Covers: Multi-Algorithm Testing, Evaluation, Hyperparameter Optimization, Feature Engineering, Interpretation, Production Pipeline

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# --- Data Preparation ---
data = pd.DataFrame({
    'feature1': np.random.randn(200),
    'feature2': np.random.randn(200),
    'target': np.random.randint(0, 2, 200)
})
X = data[['feature1', 'feature2']]
y = data['target']

# --- Algorithms to Compare ---
models = {
    'LogisticRegression': LogisticRegression(),
    'RandomForest': RandomForestClassifier(),
    'SVC': SVC(probability=True)
}

# --- Evaluation & Comparison ---
results = {}
for name, model in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
    results[name] = scores
    print(f'{name} mean accuracy: {scores.mean():.2f}')

# --- Hyperparameter Optimization ---
gs = GridSearchCV(RandomForestClassifier(), {'n_estimators': [10, 50, 100]}, cv=3)
gs.fit(X, y)
print('Best RF params:', gs.best_params_)

# --- Feature Engineering ---
X['feature_sum'] = X['feature1'] + X['feature2']

# --- Model Interpretation ---
rf = RandomForestClassifier().fit(X, y)
plt.bar(range(len(rf.feature_importances_)), rf.feature_importances_)
plt.title('Feature Importances')
plt.show()

# --- Production Pipeline Simulation ---
def predict_api(input_features):
    model = LogisticRegression().fit(X, y)
    return model.predict([input_features])

print('API prediction example:', predict_api([0.5, -0.2, 0.3]))
