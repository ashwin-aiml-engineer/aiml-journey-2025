# Project Topic 1: Comprehensive Linear Regression Analyzer
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Automated OLS regression
X = np.random.rand(30, 3)
y = np.random.rand(30)
model = LinearRegression().fit(X, y)
pred = model.predict(X)
print("OLS R2:", r2_score(y, pred))
print("OLS MSE:", mean_squared_error(y, pred))

# Assumption testing: VIF
vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
print("VIF:", vif)

# Residual analysis
residuals = y - pred
print("Residuals mean:", np.mean(residuals))

# Project Topic 2: Regularization Optimizer
ridge = Ridge(alpha=1.0).fit(X, y)
lasso = Lasso(alpha=0.1).fit(X, y)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X, y)
print("Ridge R2:", r2_score(y, ridge.predict(X)))
print("Lasso R2:", r2_score(y, lasso.predict(X)))
print("ElasticNet R2:", r2_score(y, elastic.predict(X)))

# Project Topic 3: Advanced Logistic Regression Suite
X_log = np.random.rand(40, 2)
y_log = np.random.randint(0, 2, 40)
log_model = LogisticRegression().fit(X_log, y_log)
log_pred = log_model.predict(X_log)
print("Confusion Matrix:\n", confusion_matrix(y_log, log_pred))
print("ROC AUC:", roc_auc_score(y_log, log_model.predict_proba(X_log)[:,1]))

# Project Topic 4: Model Comparison Framework
models = [LinearRegression(), Ridge(), Lasso(), ElasticNet()]
for m in models:
    scores = cross_val_score(m, X, y, cv=3)
    print(f"{m.__class__.__name__} CV mean:", np.mean(scores))

# Project Topic 5: Feature Engineering Pipeline for Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model_poly = LinearRegression().fit(X_poly, y)
print("Poly R2:", r2_score(y, model_poly.predict(X_poly)))

# Project Topic 6: Production-Ready Regression System
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model_prod = Ridge().fit(X_train, y_train)
pred_prod = model_prod.predict(X_test)
print("Production Ridge R2:", r2_score(y_test, pred_prod))
# Model serialization (simulation)
import joblib
joblib.dump(model_prod, 'ridge_model.joblib')
print("Model saved as ridge_model.joblib")

# Test all components
if __name__ == "__main__":
    print("\n--- Comprehensive Linear Regression Analyzer Test ---")
    X = np.random.rand(30, 3)
    y = np.random.rand(30)
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    print("OLS R2:", r2_score(y, pred))
    print("OLS MSE:", mean_squared_error(y, pred))
    vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    print("VIF:", vif)
    residuals = y - pred
    print("Residuals mean:", np.mean(residuals))

    print("\n--- Regularization Optimizer Test ---")
    ridge = Ridge(alpha=1.0).fit(X, y)
    lasso = Lasso(alpha=0.1).fit(X, y)
    elastic = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X, y)
    print("Ridge R2:", r2_score(y, ridge.predict(X)))
    print("Lasso R2:", r2_score(y, lasso.predict(X)))
    print("ElasticNet R2:", r2_score(y, elastic.predict(X)))

    print("\n--- Advanced Logistic Regression Suite Test ---")
    X_log = np.random.rand(40, 2)
    y_log = np.random.randint(0, 2, 40)
    log_model = LogisticRegression().fit(X_log, y_log)
    log_pred = log_model.predict(X_log)
    print("Confusion Matrix:\n", confusion_matrix(y_log, log_pred))
    print("ROC AUC:", roc_auc_score(y_log, log_model.predict_proba(X_log)[:,1]))

    print("\n--- Model Comparison Framework Test ---")
    models = [LinearRegression(), Ridge(), Lasso(), ElasticNet()]
    for m in models:
        scores = cross_val_score(m, X, y, cv=3)
        print(f"{m.__class__.__name__} CV mean:", np.mean(scores))

    print("\n--- Feature Engineering Pipeline for Regression Test ---")
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model_poly = LinearRegression().fit(X_poly, y)
    print("Poly R2:", r2_score(y, model_poly.predict(X_poly)))

    print("\n--- Production-Ready Regression System Test ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model_prod = Ridge().fit(X_train, y_train)
    pred_prod = model_prod.predict(X_test)
    print("Production Ridge R2:", r2_score(y_test, pred_prod))
    import joblib
    joblib.dump(model_prod, 'ridge_model.joblib')
    print("Model saved as ridge_model.joblib")
