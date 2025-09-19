"""
Day 12: Regression Mastery (Concise Daily Practice)
Essential code patterns for AIML job prep & business use
"""

# Topic 1: Linear Regression Mathematical Foundations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Generate sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# OLS Linear Regression
model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("R2:", r2_score(y, pred))
print("MSE:", mean_squared_error(y, pred))

# Residuals
residuals = y - pred
print("Residuals:", residuals)

# Topic 2: Multiple Linear Regression & Diagnostics
X_multi = np.random.rand(10, 3)
y_multi = np.random.rand(10)
model_multi = LinearRegression().fit(X_multi, y_multi)
print("Multi Coefficients:", model_multi.coef_)

# Multicollinearity (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X_multi, i) for i in range(X_multi.shape[1])]
print("VIF:", vif)

# Topic 3: Regularized Linear Regression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
ridge = Ridge(alpha=1.0).fit(X_multi, y_multi)
lasso = Lasso(alpha=0.1).fit(X_multi, y_multi)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_multi, y_multi)
print("Ridge Coef:", ridge.coef_)
print("Lasso Coef:", lasso.coef_)
print("ElasticNet Coef:", elastic.coef_)

# Topic 4: Logistic Regression Fundamentals
from sklearn.linear_model import LogisticRegression
X_log = np.random.rand(20, 2)
y_log = np.random.randint(0, 2, 20)
log_model = LogisticRegression().fit(X_log, y_log)
log_pred = log_model.predict(X_log)
print("Logistic Coef:", log_model.coef_)

# Topic 5: Advanced Logistic Regression
log_multi = LogisticRegression(multi_class='multinomial').fit(X_log, y_log)
print("Multinomial Coef:", log_multi.coef_)

# Topic 6: Model Diagnostics & Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_multi, y_multi, cv=3)
print("Cross-val scores:", scores)

# Topic 7: Advanced Regression Techniques
from sklearn.linear_model import HuberRegressor
huber = HuberRegressor().fit(X_multi, y_multi)
print("Huber Coef:", huber.coef_)

# Topic 8: Feature Engineering for Regression
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_multi)
print("Poly Features Shape:", X_poly.shape)

# Test all components
if __name__ == "__main__":
	print("\n--- Linear Regression Test ---")
	X = np.array([[1], [2], [3], [4], [5]])
	y = np.array([2, 4, 5, 4, 5])
	model = LinearRegression().fit(X, y)
	pred = model.predict(X)
	print("Coefficients:", model.coef_)
	print("Intercept:", model.intercept_)
	print("R2:", r2_score(y, pred))
	print("MSE:", mean_squared_error(y, pred))
	print("Residuals:", y - pred)

	print("\n--- Multiple Linear Regression & VIF Test ---")
	X_multi = np.random.rand(10, 3)
	y_multi = np.random.rand(10)
	model_multi = LinearRegression().fit(X_multi, y_multi)
	print("Multi Coefficients:", model_multi.coef_)
	vif = [variance_inflation_factor(X_multi, i) for i in range(X_multi.shape[1])]
	print("VIF:", vif)

	print("\n--- Regularized Regression Test ---")
	ridge = Ridge(alpha=1.0).fit(X_multi, y_multi)
	lasso = Lasso(alpha=0.1).fit(X_multi, y_multi)
	elastic = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_multi, y_multi)
	print("Ridge Coef:", ridge.coef_)
	print("Lasso Coef:", lasso.coef_)
	print("ElasticNet Coef:", elastic.coef_)

	print("\n--- Logistic Regression Test ---")
	X_log = np.random.rand(20, 2)
	y_log = np.random.randint(0, 2, 20)
	log_model = LogisticRegression().fit(X_log, y_log)
	print("Logistic Coef:", log_model.coef_)
	log_multi = LogisticRegression(multi_class='multinomial').fit(X_log, y_log)
	print("Multinomial Coef:", log_multi.coef_)

	print("\n--- Model Diagnostics & Validation Test ---")
	scores = cross_val_score(model, X_multi, y_multi, cv=3)
	print("Cross-val scores:", scores)

	print("\n--- Advanced Regression Techniques Test ---")
	huber = HuberRegressor().fit(X_multi, y_multi)
	print("Huber Coef:", huber.coef_)

	print("\n--- Feature Engineering for Regression Test ---")
	poly = PolynomialFeatures(degree=2)
	X_poly = poly.fit_transform(X_multi)
	print("Poly Features Shape:", X_poly.shape)
