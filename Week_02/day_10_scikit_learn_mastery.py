# Day 10: Scikit-learn Mastery (Concise, Interview & Business Ready)
# Covers: Ecosystem, Preprocessing, Model Selection, Classification, Regression, Clustering, Dimensionality Reduction, Interpretation

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# --- Data Preprocessing Example ---
data = pd.DataFrame({
    'num': np.random.randn(100),
    'cat': np.random.choice(['A', 'B', 'C'], 100),
    'target': np.random.randint(0, 2, 100)
})
scaler = StandardScaler()
data['num_scaled'] = scaler.fit_transform(data[['num']])
encoder = OneHotEncoder(sparse=False)
cat_encoded = encoder.fit_transform(data[['cat']])

# --- Feature Selection ---
X = np.hstack([data[['num_scaled']].values, cat_encoded])
y = data['target']
selector = SelectKBest(k=2)
X_selected = selector.fit_transform(X, y)

# --- Pipeline Example ---
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])
pipeline.fit(X_selected, y)

# --- Model Selection & Cross-Validation ---
kf = KFold(n_splits=5)
for train_idx, test_idx in kf.split(X_selected):
    X_train, X_test = X_selected[train_idx], X_selected[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    pipeline.fit(X_train, y_train)
    print('Fold accuracy:', accuracy_score(y_test, pipeline.predict(X_test)))

# --- Grid Search ---
gs = GridSearchCV(LogisticRegression(), {'C': [0.1, 1, 10]}, cv=3)
gs.fit(X_selected, y)
print('Best C:', gs.best_params_)

# --- Classification Example ---
clf = RandomForestClassifier()
clf.fit(X_selected, y)
print('Feature importances:', clf.feature_importances_)

# --- Regression Example ---
reg = Ridge()
reg.fit(X_selected, y)
print('Regression coef:', reg.coef_)

# --- Clustering Example ---
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_selected)
print('Cluster centers:', kmeans.cluster_centers_)

# --- Dimensionality Reduction ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_selected)
print('Explained variance:', pca.explained_variance_ratio_)

# --- Model Interpretation ---
perm_importance = permutation_importance(clf, X_selected, y)
print('Permutation importances:', perm_importance.importances_mean)
