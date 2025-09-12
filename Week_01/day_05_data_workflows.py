import numpy as np
import pandas as pd
import warnings
import time
from typing import Tuple, List, Dict, Optional
import gc  # Garbage collection for memory management
import json
import pickle
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# =============================================================================
# 1. EFFICIENT DATA PIPELINE ARCHITECTURE
# =============================================================================
print("\n1. BUILDING EFFICIENT DATA PIPELINES")
print("-" * 50)

class DataPipeline:
    """
    A comprehensive data processing pipeline that combines NumPy and Pandas
    for efficient data transformation and preparation.
    """
    def __init__(self, name: str = "ML_Pipeline"):
        self.name = name
        self.steps = []
        self.data_history = []
        self.memory_usage = []
        
    def add_step(self, step_name: str, function, **kwargs):
        """Add a processing step to the pipeline"""
        self.steps.append({
            'name': step_name,
            'function': function,
            'params': kwargs
        })
        
    def execute(self, data):
        """Execute all pipeline steps sequentially"""
        print(f"\nExecuting Pipeline: {self.name}")
        print("-" * 40)
        current_data = data.copy()
        for i, step in enumerate(self.steps):
            start_time = time.time()
            print(f"Step {i+1}: {step['name']}")
            current_data = step['function'](current_data, **step['params'])
            execution_time = time.time() - start_time
            memory_usage = current_data.memory_usage(deep=True).sum() / (1024**2)  # MB
            print(f"  ✓ Completed in {execution_time:.3f}s")
            print(f"  ✓ Memory usage: {memory_usage:.2f} MB")
            print(f"  ✓ Shape: {current_data.shape}")
            self.data_history.append(current_data.copy())
            self.memory_usage.append(memory_usage)
        return current_data

# Create sample dataset for pipeline demonstration
np.random.seed(42)
sample_data = pd.DataFrame({
    'customer_id': range(1, 10001),
    'age': np.random.randint(18, 80, 10000),
    'income': np.random.normal(50000, 15000, 10000),
    'purchase_amount': np.random.exponential(100, 10000),
    'days_since_last_purchase': np.random.poisson(30, 10000),
    'category': np.random.choice(['A', 'B', 'C', 'D'], 10000),
    'is_premium': np.random.choice([True, False], 10000, p=[0.3, 0.7]),
    'satisfaction_score': np.random.normal(7.5, 1.5, 10000),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 10000)
})

# Add some missing values and outliers for realistic data
sample_data.loc[np.random.choice(10000, 500, replace=False), 'income'] = np.nan
sample_data.loc[np.random.choice(10000, 200, replace=False), 'satisfaction_score'] = np.nan
sample_data.loc[np.random.choice(10000, 50, replace=False), 'purchase_amount'] = \
    sample_data['purchase_amount'] * 20  # Create outliers

print(f"Created sample dataset with shape: {sample_data.shape}")
print(f"Initial memory usage: {sample_data.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

# =============================================================================
# 2. DATA VALIDATION AND QUALITY CHECKS
# =============================================================================
print("\n2. DATA VALIDATION AND QUALITY ASSESSMENT")
print("-" * 50)

def data_quality_check(df: pd.DataFrame) -> Dict:
    """
    Comprehensive data quality assessment using NumPy and Pandas
    """
    quality_report = {}
    quality_report['shape'] = df.shape
    quality_report['memory_usage_mb'] = df.memory_usage(deep=True).sum() / (1024**2)
    missing_data = df.isnull().sum()
    quality_report['missing_values'] = missing_data[missing_data > 0].to_dict()
    quality_report['missing_percentage'] = (missing_data / len(df) * 100).round(2).to_dict()
    quality_report['data_types'] = df.dtypes.value_counts().to_dict()
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        numerical_data = df[numerical_cols].values
        quality_report['numerical_stats'] = {
            'columns': list(numerical_cols),
            'mean': np.nanmean(numerical_data, axis=0).tolist(),
            'std': np.nanstd(numerical_data, axis=0).tolist(),
            'min': np.nanmin(numerical_data, axis=0).tolist(),
            'max': np.nanmax(numerical_data, axis=0).tolist()
        }
        outliers = {}
        for col in numerical_cols:
            col_data = df[col].dropna()
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outlier_condition = (col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))
            outliers[col] = outlier_condition.sum()
        quality_report['outliers'] = outliers
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
    if len(categorical_cols) > 0:
        quality_report['categorical_stats'] = {}
        for col in categorical_cols:
            quality_report['categorical_stats'][col] = {
                'unique_values': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'frequency': df[col].value_counts().head(5).to_dict()
            }
    return quality_report

quality_report = data_quality_check(sample_data)
print("DATA QUALITY REPORT:")
print(f"Dataset Shape: {quality_report['shape']}")
print(f"Memory Usage: {quality_report['memory_usage_mb']:.2f} MB")
print("\nMissing Values:")
for col, count in quality_report['missing_values'].items():
    percentage = quality_report['missing_percentage'][col]
    print(f"  {col}: {count} ({percentage}%)")
print("\nOutliers Detected:")
for col, count in quality_report['outliers'].items():
    print(f"  {col}: {count} outliers")

# =============================================================================
# 3. MEMORY OPTIMIZATION TECHNIQUES
# =============================================================================
print("\n3. MEMORY OPTIMIZATION STRATEGIES")
print("-" * 50)

def optimize_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by converting data types
    """
    start_memory = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"Initial memory usage: {start_memory:.2f} MB")
    optimized_df = df.copy()
    for col in optimized_df.select_dtypes(include=['int']).columns:
        col_min = optimized_df[col].min()
        col_max = optimized_df[col].max()
        if col_min >= 0:
            if col_max < 255:
                optimized_df[col] = optimized_df[col].astype(np.uint8)
            elif col_max < 65535:
                optimized_df[col] = optimized_df[col].astype(np.uint16)
            elif col_max < 4294967295:
                optimized_df[col] = optimized_df[col].astype(np.uint32)
        else:
            if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                optimized_df[col] = optimized_df[col].astype(np.int8)
            elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                optimized_df[col] = optimized_df[col].astype(np.int16)
            elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                optimized_df[col] = optimized_df[col].astype(np.int32)
    for col in optimized_df.select_dtypes(include=['float']).columns:
        optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
    for col in optimized_df.select_dtypes(include=['object']).columns:
        num_unique_values = len(optimized_df[col].unique())
        num_total_values = len(optimized_df[col])
        if num_unique_values / num_total_values < 0.5:
            optimized_df[col] = optimized_df[col].astype('category')
    end_memory = optimized_df.memory_usage(deep=True).sum() / (1024**2)
    reduction = (start_memory - end_memory) / start_memory * 100
    print(f"Optimized memory usage: {end_memory:.2f} MB")
    print(f"Memory reduction: {reduction:.1f}%")
    return optimized_df

optimized_sample_data = optimize_memory_usage(sample_data)

# =============================================================================
# 4. ADVANCED DATA CLEANING PIPELINE STEPS
# =============================================================================
print("\n4. BUILDING DATA CLEANING PIPELINE STEPS")
print("-" * 50)

def remove_outliers(df: pd.DataFrame, columns: List[str] = None, method: str = 'iqr') -> pd.DataFrame:
    """Remove outliers using specified method"""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    cleaned_df = df.copy()
    outliers_removed = 0
    for col in columns:
        if col in cleaned_df.columns:
            if method == 'iqr':
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_condition = (cleaned_df[col] < (Q1 - 1.5 * IQR)) | (cleaned_df[col] > (Q3 + 1.5 * IQR))
                outliers_count = outlier_condition.sum()
                cleaned_df = cleaned_df[~outlier_condition]
                outliers_removed += outliers_count
            elif method == 'zscore':
                z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
                outlier_condition = z_scores > 3
                outliers_count = outlier_condition.sum()
                cleaned_df = cleaned_df[~outlier_condition]
                outliers_removed += outliers_count
    print(f"  Removed {outliers_removed} outliers using {method} method")
    return cleaned_df

def handle_missing_values(df: pd.DataFrame, strategy: str = 'smart') -> pd.DataFrame:
    """Handle missing values with different strategies"""
    cleaned_df = df.copy()
    if strategy == 'smart':
        for col in cleaned_df.columns:
            missing_pct = cleaned_df[col].isnull().sum() / len(cleaned_df)
            if missing_pct > 0:
                if missing_pct > 0.5:
                    print(f"  Dropping column '{col}' (>{missing_pct:.1%} missing)")
                    cleaned_df = cleaned_df.drop(columns=[col])
                elif cleaned_df[col].dtype in ['object', 'category']:
                    mode_value = cleaned_df[col].mode().iloc[0] if not cleaned_df[col].mode().empty else 'Unknown'
                    cleaned_df[col] = cleaned_df[col].fillna(mode_value)
                    print(f"  Filled '{col}' categorical missing values with mode: {mode_value}")
                elif cleaned_df[col].dtype in [np.number]:
                    median_value = cleaned_df[col].median()
                    cleaned_df[col] = cleaned_df[col].fillna(median_value)
                    print(f"  Filled '{col}' numerical missing values with median: {median_value:.2f}")
    return cleaned_df

def normalize_data(df: pd.DataFrame, columns: List[str] = None, method: str = 'minmax') -> pd.DataFrame:
    """Normalize numerical data"""
    normalized_df = df.copy()
    if columns is None:
        columns = normalized_df.select_dtypes(include=[np.number]).columns
    for col in columns:
        if col in normalized_df.columns:
            col_data = normalized_df[col].values
            if method == 'minmax':
                min_val = np.min(col_data)
                max_val = np.max(col_data)
                if max_val != min_val:
                    normalized_df[col] = (col_data - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = np.mean(col_data)
                std_val = np.std(col_data)
                if std_val != 0:
                    normalized_df[col] = (col_data - mean_val) / std_val
    print(f"  Applied {method} normalization to {len(columns)} columns")
    return normalized_df

# Create and execute a comprehensive data pipeline
pipeline = DataPipeline("Customer_Data_Processing_Pipeline")
pipeline.add_step("Memory Optimization", optimize_memory_usage)
pipeline.add_step("Missing Values Handling", handle_missing_values, strategy='smart')
pipeline.add_step("Outlier Removal", remove_outliers, method='iqr', columns=['income', 'purchase_amount'])
pipeline.add_step("Data Normalization", normalize_data, method='minmax', 
                 columns=['age', 'income', 'purchase_amount', 'satisfaction_score'])
processed_data = pipeline.execute(sample_data)

print(f"\n🎯 PIPELINE EXECUTION COMPLETED!")
print(f"Original shape: {sample_data.shape}")
print(f"Final shape: {processed_data.shape}")
print(f"Data reduction: {(1 - processed_data.shape[0]/sample_data.shape[0])*100:.1f}%")

# =============================================================================
# 5. ADVANCED FEATURE ENGINEERING
# =============================================================================
print("\n5. FEATURE ENGINEERING FOR MACHINE LEARNING")
print("-" * 50)

class FeatureEngineer:
    """
    Advanced feature engineering class combining NumPy and Pandas
    for creating ML-ready features
    """
    def __init__(self):
        self.feature_history = {}
        self.encoders = {}
        
    def create_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_df = df.copy()
        print("Creating Numerical Features:")
        if 'age' in feature_df.columns and 'income' in feature_df.columns:
            feature_df['age_income_interaction'] = feature_df['age'] * feature_df['income']
            feature_df['age_squared'] = np.power(feature_df['age'], 2)
            feature_df['income_log'] = np.log1p(np.abs(feature_df['income']))
            print("  ✓ Polynomial and interaction features")
        if 'age' in feature_df.columns:
            age_bins = [0, 25, 35, 50, 65, 100]
            feature_df['age_group'] = pd.cut(feature_df['age'], bins=age_bins, 
                                           labels=['Young', 'Adult', 'Middle', 'Senior', 'Elder'])
            print("  ✓ Age binning completed")
        if 'purchase_amount' in feature_df.columns:
            feature_df = feature_df.sort_values('customer_id').reset_index(drop=True)
            feature_df['purchase_amount_rolling_mean'] = feature_df['purchase_amount'].rolling(
                window=50, min_periods=1).mean()
            feature_df['purchase_amount_rolling_std'] = feature_df['purchase_amount'].rolling(
                window=50, min_periods=1).std().fillna(0)
            print("  ✓ Rolling statistical features")
        for col in ['income', 'purchase_amount', 'satisfaction_score']:
            if col in feature_df.columns:
                feature_df[f'{col}_percentile'] = feature_df[col].rank(pct=True)
                quantiles = np.percentile(feature_df[col], [25, 50, 75])
                conditions = [
                    feature_df[col] <= quantiles[0],
                    (feature_df[col] > quantiles[0]) & (feature_df[col] <= quantiles[1]),
                    (feature_df[col] > quantiles[1]) & (feature_df[col] <= quantiles[2]),
                    feature_df[col] > quantiles[2]
                ]
                choices = ['Low', 'Medium', 'High', 'Very High']
                feature_df[f'{col}_quartile'] = np.select(conditions, choices, default='Unknown')
        print("  ✓ Percentile and quartile features")
        return feature_df
    
    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_df = df.copy()
        print("\nCreating Categorical Features:")
        for col in ['category', 'region']:
            if col in feature_df.columns:
                freq_map = feature_df[col].value_counts().to_dict()
                feature_df[f'{col}_frequency'] = feature_df[col].map(freq_map)
                print(f"  ✓ Frequency encoding for {col}")
        target = np.random.choice([0, 1], len(feature_df), p=[0.7, 0.3])
        feature_df['target'] = target
        for col in ['category', 'region']:
            if col in feature_df.columns:
                target_mean = feature_df.groupby(col)['target'].mean()
                feature_df[f'{col}_target_encoded'] = feature_df[col].map(target_mean)
                print(f"  ✓ Target encoding for {col}")
        categorical_cols = ['category', 'region', 'age_group']
        for col in categorical_cols:
            if col in feature_df.columns:
                dummies = pd.get_dummies(feature_df[col], prefix=f'{col}_is', drop_first=True)
                feature_df = pd.concat([feature_df, dummies], axis=1)
                self.encoders[f'{col}_dummies'] = dummies.columns.tolist()
        print("  ✓ One-hot encoding completed")
        ordinal_cols = ['income_quartile', 'purchase_amount_quartile', 'satisfaction_score_quartile']
        for col in ordinal_cols:
            if col in feature_df.columns:
                le = LabelEncoder()
                feature_df[f'{col}_encoded'] = le.fit_transform(feature_df[col].astype(str))
                self.encoders[f'{col}_label_encoder'] = le
        print("  ✓ Label encoding for ordinal features")
        return feature_df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_df = df.copy()
        print("\nCreating Time-based Features:")
        start_date = datetime(2024, 1, 1)
        feature_df['last_purchase_date'] = [
            start_date + timedelta(days=int(days)) 
            for days in feature_df['days_since_last_purchase']
        ]
        feature_df['purchase_month'] = feature_df['last_purchase_date'].dt.month
        feature_df['purchase_quarter'] = feature_df['last_purchase_date'].dt.quarter
        feature_df['purchase_day_of_week'] = feature_df['last_purchase_date'].dt.dayofweek
        feature_df['is_weekend'] = feature_df['purchase_day_of_week'].isin([5, 6]).astype(int)
        feature_df['month_sin'] = np.sin(2 * np.pi * feature_df['purchase_month'] / 12)
        feature_df['month_cos'] = np.cos(2 * np.pi * feature_df['purchase_month'] / 12)
        feature_df['day_sin'] = np.sin(2 * np.pi * feature_df['purchase_day_of_week'] / 7)
        feature_df['day_cos'] = np.cos(2 * np.pi * feature_df['purchase_day_of_week'] / 7)
        print("  ✓ Time component extraction")
        print("  ✓ Cyclical encoding for temporal features")
        return feature_df

feature_engineer = FeatureEngineer()
feature_rich_data = processed_data.copy()
feature_rich_data = feature_engineer.create_numerical_features(feature_rich_data)
feature_rich_data = feature_engineer.create_categorical_features(feature_rich_data)
feature_rich_data = feature_engineer.create_time_features(feature_rich_data)

print(f"\n🎯 Feature Engineering Complete!")
print(f"Original features: {processed_data.shape[1]}")
print(f"Engineered features: {feature_rich_data.shape[1]}")
print(f"New features created: {feature_rich_data.shape[1] - processed_data.shape[1]}")

# =============================================================================
# 6. DATA AGGREGATION AND SUMMARIZATION
# =============================================================================
print("\n6. ADVANCED DATA AGGREGATION STRATEGIES")
print("-" * 50)

def create_customer_summary(df: pd.DataFrame) -> pd.DataFrame:
    print("Creating Customer Aggregations:")
    agg_functions = {
        'purchase_amount': ['mean', 'sum', 'std', 'count', 'min', 'max'],
        'satisfaction_score': ['mean', 'std'],
        'days_since_last_purchase': ['mean', 'min'],
        'age': 'first',
        'income': 'first',
        'is_premium': 'first'
    }
    customer_summary = df.groupby(['category', 'region']).agg(agg_functions).round(3)
    customer_summary.columns = ['_'.join(col).strip() if col[1] else col[0] 
                               for col in customer_summary.columns.values]
    customer_summary['avg_purchase_per_day'] = (
        customer_summary['purchase_amount_sum'] / 
        customer_summary['days_since_last_purchase_mean']
    ).round(3)
    customer_summary['satisfaction_to_purchase_ratio'] = (
        customer_summary['satisfaction_score_mean'] / 
        customer_summary['purchase_amount_mean']
    ).round(3)
    print(f"  ✓ Created {len(customer_summary)} customer segments")
    return customer_summary.reset_index()

def create_pivot_analysis(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    print("\nCreating Pivot Analyses:")
    pivot_tables = {}
    pivot_tables['category_region'] = pd.pivot_table(
        df, 
        values=['purchase_amount', 'satisfaction_score'], 
        index='category', 
        columns='region',
        aggfunc='mean',
        fill_value=0
    ).round(2)
    if 'age_group' in df.columns:
        pivot_tables['age_group_performance'] = pd.pivot_table(
            df,
            values=['purchase_amount', 'income', 'satisfaction_score'],
            index='age_group',
            aggfunc=['mean', 'count'],
            fill_value=0
        ).round(2)
    if 'purchase_quarter' in df.columns:
        pivot_tables['quarterly_trends'] = pd.pivot_table(
            df,
            values=['purchase_amount', 'satisfaction_score'],
            index='purchase_quarter',
            columns='is_premium',
            aggfunc='mean',
            fill_value=0
        ).round(2)
    print(f"  ✓ Created {len(pivot_tables)} pivot analyses")
    return pivot_tables

customer_summary = create_customer_summary(feature_rich_data)
pivot_analyses = create_pivot_analysis(feature_rich_data)

print("\nSample Customer Summary:")
print(customer_summary.head())
print("\nCategory vs Region Pivot:")
print(pivot_analyses['category_region'])

# =============================================================================
# 7. ML DATA PREPARATION WORKFLOWS
# =============================================================================
print("\n7. MACHINE LEARNING DATA PREPARATION")
print("-" * 50)

class MLDataPreparator:
    """
    Complete ML data preparation pipeline
    """
    def __init__(self):
        self.feature_columns = None
        self.target_column = None
        self.preprocessing_stats = {}
        
    def prepare_ml_dataset(self, df: pd.DataFrame, target_col: str = 'target') -> Dict:
        print("Preparing ML Dataset:")
        if target_col in df.columns:
            X = df.drop(columns=[target_col, 'customer_id', 'last_purchase_date'], errors='ignore')
            y = df[target_col]
        else:
            X = df.drop(columns=['customer_id', 'last_purchase_date'], errors='ignore')
            y = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_columns]
        X_numeric = X_numeric.fillna(X_numeric.mean())
        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y, test_size=0.2, random_state=42, stratify=y
        )
        self.feature_columns = list(X_numeric.columns)
        self.preprocessing_stats = {
            'total_samples': len(df),
            'total_features': len(X_numeric.columns),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'positive_class_ratio': np.mean(y),
            'feature_dtypes': X_numeric.dtypes.value_counts().to_dict()
        }
        print(f"  ✓ Dataset split: {len(X_train)} train, {len(X_test)} test samples")
        print(f"  ✓ Features selected: {len(X_numeric.columns)}")
        print(f"  ✓ Target balance: {np.mean(y):.2%} positive class")
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_columns,
            'preprocessing_stats': self.preprocessing_stats
        }
    
    def create_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        print("\nCreating Optimized Feature Matrix:")
        numeric_features = df.select_dtypes(include=[np.number])
        feature_matrix = numeric_features.values
        col_means = np.nanmean(feature_matrix, axis=0)
        nan_indices = np.where(np.isnan(feature_matrix))
        feature_matrix[nan_indices] = np.take(col_means, nan_indices[1])
        means = np.mean(feature_matrix, axis=0)
        stds = np.std(feature_matrix, axis=0)
        stds[stds == 0] = 1
        feature_matrix_normalized = (feature_matrix - means) / stds
        print(f"  ✓ Feature matrix shape: {feature_matrix_normalized.shape}")
        print(f"  ✓ Memory usage: {feature_matrix_normalized.nbytes / (1024**2):.2f} MB")
        return feature_matrix_normalized
    
    def validate_ml_readiness(self, ml_data: Dict) -> Dict:
        print("\nValidating ML Readiness:")
        validation_results = {
            'is_ready': True,
            'issues': [],
            'recommendations': []
        }
        X_train, y_train = ml_data['X_train'], ml_data['y_train']
        correlations = []
        for col in X_train.columns:
            corr = np.corrcoef(X_train[col], y_train)[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
        max_correlation = max(correlations) if correlations else 0
        if max_correlation > 0.95:
            validation_results['issues'].append("Possible data leakage detected")
            validation_results['is_ready'] = False
        class_distribution = np.bincount(y_train) / len(y_train)
        min_class_ratio = min(class_distribution)
        if min_class_ratio < 0.05:
            validation_results['issues'].append("Severe class imbalance detected")
            validation_results['recommendations'].append("Consider SMOTE or class weighting")
        low_variance_features = []
        for col in X_train.columns:
            if X_train[col].std() < 0.01:
                low_variance_features.append(col)
        if low_variance_features:
            validation_results['issues'].append(f"Low variance features: {len(low_variance_features)}")
            validation_results['recommendations'].append("Consider removing low variance features")
        if len(X_train.columns) > 1:
            correlation_matrix = X_train.corr().abs()
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if correlation_matrix.iloc[i, j] > 0.9:
                        high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
            if high_corr_pairs:
                validation_results['issues'].append(f"High multicollinearity: {len(high_corr_pairs)} pairs")
                validation_results['recommendations'].append("Consider PCA or feature selection")
        print(f"  ✓ Data validation: {'PASSED' if validation_results['is_ready'] else 'ISSUES FOUND'}")
        if validation_results['issues']:
            for issue in validation_results['issues']:
                print(f"    ⚠ {issue}")
        return validation_results

ml_preparator = MLDataPreparator()
ml_data = ml_preparator.prepare_ml_dataset(feature_rich_data, 'target')
feature_matrix = ml_preparator.create_feature_matrix(feature_rich_data)
validation_results = ml_preparator.validate_ml_readiness(ml_data)

# =============================================================================
# 8. EXPORT AND INTEGRATION STRATEGIES
# =============================================================================
print("\n8. DATA EXPORT AND INTEGRATION")
print("-" * 50)

class DataExporter:
    """
    Export processed data in various formats for different use cases
    """
    def __init__(self, base_filename: str = "processed_data"):
        self.base_filename = base_filename
        self.export_log = []
    
    def export_to_csv(self, df: pd.DataFrame, suffix: str = "") -> str:
        filename = f"{self.base_filename}{suffix}.csv"
        export_df = df.copy()
        for col in export_df.select_dtypes(include=['datetime64']):
            export_df[col] = export_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        export_df.to_csv(filename, index=False, compression='gzip')
        file_size = len(export_df.to_csv(index=False).encode()) / (1024**2)
        self.export_log.append({
            'format': 'CSV',
            'filename': filename,
            'size_mb': file_size,
            'rows': len(export_df),
            'columns': len(export_df.columns)
        })
        print(f"  ✓ Exported to CSV: {filename} ({file_size:.2f} MB)")
        return filename
    
    def export_to_parquet(self, df: pd.DataFrame, suffix: str = "") -> str:
        filename = f"{self.base_filename}{suffix}.parquet"
        try:
            df.to_parquet(filename, compression='snappy')
            file_size = len(df) * len(df.columns) * 8 / (1024**2)
            self.export_log.append({
                'format': 'Parquet',
                'filename': filename,
                'size_mb': file_size,
                'rows': len(df),
                'columns': len(df.columns)
            })
            print(f"  ✓ Exported to Parquet: {filename} (~{file_size:.2f} MB)")
            return filename
        except ImportError:
            print("  ⚠ Parquet export requires pyarrow or fastparquet")
            return None
    
    def export_ml_ready_data(self, ml_data: Dict) -> Dict[str, str]:
        exported_files = {}
        train_data = pd.DataFrame(ml_data['X_train'])
        train_data['target'] = ml_data['y_train'].values
        exported_files['train'] = self.export_to_csv(train_data, "_train")
        test_data = pd.DataFrame(ml_data['X_test'])
        test_data['target'] = ml_data['y_test'].values
        exported_files['test'] = self.export_to_csv(test_data, "_test")
        feature_metadata = {
            'feature_names': ml_data['feature_names'],
            'preprocessing_stats': ml_data['preprocessing_stats'],
            'export_timestamp': datetime.now().isoformat()
        }

        # Convert feature_dtypes keys to str for JSON compatibility
        if 'feature_dtypes' in feature_metadata['preprocessing_stats']:
            feature_metadata['preprocessing_stats']['feature_dtypes'] = {
                str(k): v for k, v in feature_metadata['preprocessing_stats']['feature_dtypes'].items()
            }
        metadata_filename = f"{self.base_filename}_metadata.json"
        with open(metadata_filename, 'w') as f:
            json.dump(feature_metadata, f, indent=2, default=str)
        exported_files['metadata'] = metadata_filename
        print(f"  ✓ Exported ML metadata: {metadata_filename}")
        return exported_files
    
    def export_feature_engineered_data(self, df: pd.DataFrame) -> str:
        return self.export_to_csv(df, "_feature_engineered")
    
    def save_pipeline_objects(self, feature_engineer: FeatureEngineer, 
                            ml_preparator: MLDataPreparator) -> str:
        pipeline_objects = {
            'encoders': feature_engineer.encoders,
            'feature_columns': ml_preparator.feature_columns,
            'preprocessing_stats': ml_preparator.preprocessing_stats,
            'creation_timestamp': datetime.now().isoformat()
        }
        filename = f"{self.base_filename}_pipeline.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(pipeline_objects, f)
        print(f"  ✓ Saved pipeline objects: {filename}")
        return filename
    
    def generate_export_report(self) -> pd.DataFrame:
        if not self.export_log:
            return pd.DataFrame()
        report_df = pd.DataFrame(self.export_log)
        total_size = report_df['size_mb'].sum()
        print(f"\n📊 EXPORT SUMMARY:")
        print(f"Total files exported: {len(report_df)}")
        print(f"Total size: {total_size:.2f} MB")
        print("\nFile breakdown:")
        for _, row in report_df.iterrows():
            print(f"  {row['format']}: {row['filename']} ({row['size_mb']:.2f} MB)")
        return report_df

print("\nExecuting Export Workflow:")
exporter = DataExporter("day_05_customer_analysis")
exporter.export_feature_engineered_data(feature_rich_data)
ml_files = exporter.export_ml_ready_data(ml_data)
exporter.export_to_csv(customer_summary, "_customer_summary")
exporter.save_pipeline_objects(feature_engineer, ml_preparator)
export_report = exporter.generate_export_report()

print("\n" + "="*60)
print("COMPLETE DATA WORKFLOWS SUMMARY")
print("="*60)
print(f"\n📊 FINAL STATISTICS:")
print(f"Original dataset: {processed_data.shape}")
print(f"Feature-engineered dataset: {feature_rich_data.shape}")
print(f"ML-ready features: {len(ml_data['feature_names'])}")
print(f"Export files created: {len(exporter.export_log)}")
print("="*60)