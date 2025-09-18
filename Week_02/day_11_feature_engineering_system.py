# Project Topic 1: Intelligent Data Cleaning System
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

def automated_data_cleaning(df):
    # Missing value detection & treatment
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    # Outlier identification (Isolation Forest)
    iso = IsolationForest()
    outliers = iso.fit_predict(df_imputed)
    df_clean = df_imputed[outliers == 1]
    # Data quality assessment
    report = df_clean.isnull().sum().to_dict()
    # Memory optimization
    mem_usage = df_clean.memory_usage(deep=True).sum() / 1024
    return df_clean, report, mem_usage

# Project Topic 2: Smart Feature Scaling Engine
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def smart_scaling(df):
    # Automatic method selection
    if abs(df.skew().mean()) > 1:
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled, scaler.__class__.__name__

# Project Topic 3: Advanced Categorical Encoder
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

def advanced_categorical_encoding(df, col):
    # Multi-strategy encoding
    if df[col].nunique() > 10:
        encoder = OrdinalEncoder()
    else:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[[col]])
    return encoded, encoder.__class__.__name__

# Project Topic 4: Feature Selection Optimizer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression

def feature_selection_optimizer(X, y):
    # Multi-method comparison
    kbest = SelectKBest(score_func=f_classif, k=2).fit(X, y)
    rfe = RFE(LogisticRegression(), n_features_to_select=2).fit(X, y)
    # Importance ranking
    kbest_scores = kbest.scores_
    rfe_ranking = rfe.ranking_
    return kbest_scores, rfe_ranking

# Project Topic 5: Complete Preprocessing Pipeline
from sklearn.pipeline import Pipeline

def complete_preprocessing_pipeline():
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    return pipe

# Project Topic 6: Feature Engineering Laboratory

def feature_engineering_lab(df):
    # Automated feature creation
    df['sum'] = df.sum(axis=1)
    df['log_A'] = np.log1p(df.iloc[:, 0])
    df['A_times_B'] = df.iloc[:, 0] * df.iloc[:, 1]
    # Mathematical transformation exploration
    df['sqrt_A'] = np.sqrt(df.iloc[:, 0])
    # Feature interaction discovery
    df['interaction'] = df.iloc[:, 0] / (df.iloc[:, 1] + 1)
    return df

# Testing the functions
if __name__ == "__main__":
    print("\n--- Testing Intelligent Data Cleaning System ---")
    df = pd.DataFrame({
        'A': [1, np.nan, 3, 100],
        'B': [4, 5, np.nan, 200],
        'C': [7, 8, 9, 300]
    })
    clean_df, report, mem_usage = automated_data_cleaning(df)
    print("Cleaned DataFrame:\n", clean_df)
    print("Missing Value Report:", report)
    print("Memory Usage (KB):", mem_usage)

    print("\n--- Testing Smart Feature Scaling Engine ---")
    scaled_df, scaler_name = smart_scaling(clean_df)
    print("Scaled DataFrame:\n", scaled_df)
    print("Scaler Used:", scaler_name)

    print("\n--- Testing Advanced Categorical Encoder ---")
    cat_df = pd.DataFrame({'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue', 'Green', 'Yellow', 'Purple', 'Orange', 'Black', 'White']})
    encoded, encoder_name = advanced_categorical_encoding(cat_df, 'Color')
    print("Encoded Data:\n", encoded)
    print("Encoder Used:", encoder_name)

    print("\n--- Testing Feature Selection Optimizer ---")
    X = np.random.rand(10, 3)
    y = np.random.randint(0, 2, 10)
    kbest_scores, rfe_ranking = feature_selection_optimizer(X, y)
    print("SelectKBest Scores:", kbest_scores)
    print("RFE Ranking:", rfe_ranking)

    print("\n--- Testing Complete Preprocessing Pipeline ---")
    pipe = complete_preprocessing_pipeline()
    pipe_result = pipe.fit_transform(df)
    print("Pipeline Output:\n", pipe_result)

    print("\n--- Testing Feature Engineering Laboratory ---")
    fe_df = feature_engineering_lab(clean_df.copy())
    print("Feature Engineered DataFrame:\n", fe_df)
