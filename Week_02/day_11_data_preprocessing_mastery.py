# 1. Missing Value Imputation
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [4, 5, np.nan]})
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# 2. Outlier Detection & Handling
from sklearn.ensemble import IsolationForest
iso = IsolationForest()
outliers = iso.fit_predict(df_imputed)
df_no_outliers = df_imputed[outliers == 1]

# 3. Feature Scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_no_outliers), columns=df_no_outliers.columns)

# 4. Encoding Categorical Variables
cat_df = pd.DataFrame({'Color': ['Red', 'Blue', 'Green']})
cat_encoded = pd.get_dummies(cat_df, columns=['Color'])

# 5. Feature Selection
from sklearn.feature_selection import SelectKBest, f_classif
X = np.random.rand(10, 5)
y = np.random.randint(0, 2, 10)
selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X, y)

# 6. Feature Engineering (Simple Example)
df_fe = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df_fe['A_times_B'] = df_fe['A'] * df_fe['B']

# 7. Pipeline Construction
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
df_pipe = pipe.fit_transform(df)

# 8. Cross-Validation
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
X_cv = np.random.rand(20, 3)
y_cv = np.random.randint(0, 2, 20)
model = LogisticRegression()
scores = cross_val_score(model, X_cv, y_cv, cv=3)

# 9. Data Leakage Prevention (Simple Example)
# Always split data BEFORE preprocessing
from sklearn.model_selection import train_test_split
X_leak, y_leak = df.values, np.array([0, 1, 0])
X_train, X_test, y_train, y_test = train_test_split(X_leak, y_leak, test_size=0.33, random_state=42)
scaler_leak = StandardScaler().fit(X_train)
X_train_scaled = scaler_leak.transform(X_train)
X_test_scaled = scaler_leak.transform(X_test)

# Display results for verification
if __name__ == "__main__":
    print("\n--- 1. Missing Value Imputation ---")
    df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [4, 5, np.nan]})
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    print(df_imputed)

    print("\n--- 2. Outlier Detection & Handling ---")
    iso = IsolationForest()
    outliers = iso.fit_predict(df_imputed)
    df_no_outliers = df_imputed[outliers == 1]
    print(df_no_outliers)

    print("\n--- 3. Feature Scaling ---")
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_no_outliers), columns=df_no_outliers.columns)
    print(df_scaled)

    print("\n--- 4. Encoding Categorical Variables ---")
    cat_df = pd.DataFrame({'Color': ['Red', 'Blue', 'Green']})
    cat_encoded = pd.get_dummies(cat_df, columns=['Color'])
    print(cat_encoded)

    print("\n--- 5. Feature Selection ---")
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)
    selector = SelectKBest(score_func=f_classif, k=2)
    X_selected = selector.fit_transform(X, y)
    print(X_selected)

    print("\n--- 6. Feature Engineering ---")
    df_fe = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df_fe['A_times_B'] = df_fe['A'] * df_fe['B']
    print(df_fe)

    print("\n--- 7. Pipeline Construction ---")
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    df_pipe = pipe.fit_transform(df)
    print(df_pipe)

    print("\n--- 8. Cross-Validation ---")
    X_cv = np.random.rand(20, 3)
    y_cv = np.random.randint(0, 2, 20)
    model = LogisticRegression()
    scores = cross_val_score(model, X_cv, y_cv, cv=3)
    print(scores)

    print("\n--- 9. Data Leakage Prevention ---")
    X_leak, y_leak = df.values, np.array([0, 1, 0])
    X_train, X_test, y_train, y_test = train_test_split(X_leak, y_leak, test_size=0.33, random_state=42)
    scaler_leak = StandardScaler().fit(X_train)
    X_train_scaled = scaler_leak.transform(X_train)
    X_test_scaled = scaler_leak.transform(X_test)
    print("Train Scaled:\n", X_train_scaled)
    print("Test Scaled:\n", X_test_scaled)
