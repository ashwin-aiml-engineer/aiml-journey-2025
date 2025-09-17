# Day 10 Mini Project: End-to-End ML Pipeline (Concise)
# Goal: Practice a minimal ML workflow for interviews/business

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- Data Preparation ---
data = pd.DataFrame({
    'feature': np.arange(10),
    'target': np.arange(10) * 2 + 1
})
X = data[['feature']]
y = data['target']

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training ---

# --- Advanced Preprocessing ---
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
selector = SelectKBest(score_func=f_regression, k=1)
X_selected = selector.fit_transform(X_scaled, y)

# --- Multi-Algorithm Comparison ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'RandomForest': RandomForestRegressor()
}
for name, model in models.items():
    scores = cross_val_score(model, X_selected, y, cv=3, scoring='neg_mean_squared_error')
    print(f'{name} CV MSE: {abs(scores.mean()):.2f}')

# --- Train/Test Split & Best Model Training ---
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
best_model = RandomForestRegressor()
best_model.fit(X_train, y_train)
pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, pred)
print(f'RandomForest Test MSE: {mse:.2f}')

# --- Feature Importance Visualization ---
import matplotlib.pyplot as plt
plt.bar(range(len(best_model.feature_importances_)), best_model.feature_importances_)
plt.title('Feature Importances')
plt.show()
