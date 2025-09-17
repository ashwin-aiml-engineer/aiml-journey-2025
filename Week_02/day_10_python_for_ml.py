# Day 10: Python for Machine Learning (Concise Daily Practice)
# Covers: Numpy, Pandas, Scikit-learn essentials for AIML interviews

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# --- Numpy: Array Basics ---
arr = np.array([1, 2, 3, 4])
print('Numpy Array:', arr)

# --- Pandas: DataFrame Basics ---
df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
print('Pandas DataFrame:\n', df)

# --- Scikit-learn: Simple Regression ---
X = df[['feature1']]
y = df['feature2']
model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)
print('Predictions:', pred)
