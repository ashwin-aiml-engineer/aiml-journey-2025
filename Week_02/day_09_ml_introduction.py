import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.cluster import KMeans

print("=== PRACTICE SESSION: ML BASICS ===")

# 1. What is Machine Learning?
print("\nWhat is Machine Learning?")
print("ML lets computers learn patterns from data and make predictions.")

# 2. Types of ML
print("\nTypes of Machine Learning:")
print(" - Supervised: Learn from labeled data (classification, regression)")
print(" - Unsupervised: Find patterns in unlabeled data (clustering)")
print(" - Reinforcement: Learn by trial and error (not covered here)")

# 3. ML Workflow (short version)
print("\nML Workflow: Problem -> Data -> Model -> Evaluate -> Improve")

# 4. Practice: Classification
print("\n--- Classification Example ---")
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression().fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 5. Practice: Regression
print("\n--- Regression Example ---")
X, y = make_regression(n_samples=200, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))

# 6. Practice: Clustering
print("\n--- Clustering Example ---")
X, _ = make_blobs(n_samples=150, centers=3, random_state=42)
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
print("Cluster centers:\n", kmeans.cluster_centers_)

# 7. Summary
print("\nPractice session complete! You tried classification, regression, and clustering in scikit-learn.")
