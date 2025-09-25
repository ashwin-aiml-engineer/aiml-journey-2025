"""
Day 18: Time Series Analysis (15-Minute Daily Practice)
🎯 Master time series essentials quickly
✅ Stationarity, decomposition, forecasting basics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def create_sample_data():
    """Generate sample time series"""
    print("📊 Creating time series data...")
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=500, freq='D')
    trend = np.linspace(100, 120, 500)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(500) / 365.25)
    noise = np.random.normal(0, 2, 500)
    values = trend + seasonal + noise
    
    df = pd.DataFrame({'value': values}, index=dates)
    print(f"✅ Created 500 days of data ({df['value'].min():.1f} to {df['value'].max():.1f})")
    return df

def check_stationarity(data):
    """Quick stationarity check"""
    print("\n📈 STATIONARITY CHECK")
    rolling_mean = data.rolling(30).mean()
    rolling_std = data.rolling(30).std()
    
    mean_change = abs(rolling_mean.iloc[-30:].mean() - rolling_mean.iloc[:30].mean())
    std_change = abs(rolling_std.iloc[-30:].mean() - rolling_std.iloc[:30].mean())
    
    print(f"  Mean change: {mean_change:.2f}")
    print(f"  Std change: {std_change:.2f}")
    
    if mean_change < 3 and std_change < 1:
        print("  ✅ Series is stationary")
        return True, rolling_mean
    else:
        print("  ❌ Non-stationary - need differencing")
        diff_series = data.diff().dropna()
        print(f"  After differencing: mean={diff_series.mean():.2f}")
        return False, diff_series

def decompose_series(data):
    """Simple seasonal decomposition"""
    print("\n🔍 DECOMPOSITION")
    trend = data.rolling(365, center=True).mean()
    detrended = data - trend
    seasonal = detrended.groupby(detrended.index.dayofyear).mean()
    
    # Map seasonal component
    seasonal_full = data.copy()
    for i, date in enumerate(data.index):
        seasonal_full.iloc[i] = seasonal.get(date.dayofyear, 0)
    
    residual = data - trend - seasonal_full
    
    print(f"  Trend strength: {trend.std():.2f}")
    print(f"  Seasonal strength: {seasonal_full.std():.2f}")
    print(f"  Residual noise: {residual.std():.2f}")
    
    return trend, seasonal_full, residual

def create_features(data):
    """Essential forecasting features"""
    print("\n⚙️ FEATURE ENGINEERING")
    df = pd.DataFrame({'value': data})
    
    # Lag features
    for lag in [1, 2, 3, 7]:
        df[f'lag_{lag}'] = df['value'].shift(lag)
    
    # Rolling stats
    df['ma_7'] = df['value'].rolling(7).mean()
    df['ma_30'] = df['value'].rolling(30).mean()
    
    # Time features
    df['day_week'] = df.index.dayofweek
    df['day_year'] = df.index.dayofyear
    
    feature_cols = [col for col in df.columns if col != 'value']
    df_clean = df.dropna()
    
    print(f"  Created {len(feature_cols)} features, {len(df_clean)} clean rows")
    return df_clean, feature_cols

def simple_forecast(data, periods=30):
    """Simple AR forecast"""
    print("\n🔮 FORECASTING")
    
    # Prepare AR data
    values = data.values
    X, y = [], []
    for i in range(3, len(values)):
        X.append(values[i-3:i])
        y.append(values[i])
    
    X, y = np.array(X), np.array(y)
    
    # Fit model
    model = LinearRegression().fit(X, y)
    
    # Generate forecast
    forecast = []
    last_vals = list(values[-3:])
    
    for _ in range(periods):
        pred = model.predict([last_vals[-3:]])[0]
        forecast.append(pred)
        last_vals.append(pred)
    
    forecast_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=periods)
    forecast_df = pd.DataFrame({'forecast': forecast}, index=forecast_dates)
    
    print(f"  R²: {model.score(X, y):.3f}")
    print(f"  Forecast: {min(forecast):.1f} to {max(forecast):.1f}")
    
    return forecast_df

def compare_ml_models(df_features, feature_cols):
    """Quick ML comparison"""
    print("\n🤖 ML COMPARISON")
    
    split = int(0.8 * len(df_features))
    X_train, X_test = df_features[feature_cols].iloc[:split], df_features[feature_cols].iloc[split:]
    y_train, y_test = df_features['value'].iloc[:split], df_features['value'].iloc[split:]
    
    models = {
        'Linear': LinearRegression(),
        'Random Forest': RandomForestRegressor(50, random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        print(f"  {name}: MSE = {mse:.2f}")

def create_visualization(data, forecast, trend=None):
    """Simple 2-panel visualization"""
    print("\n📊 Creating visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Recent data + forecast
    recent = data.iloc[-60:]
    ax1.plot(recent.index, recent.values, label='Actual', linewidth=2)
    ax1.plot(forecast.index, forecast['forecast'], 'r--', label='Forecast', linewidth=2)
    ax1.axvline(data.index[-1], color='gray', linestyle=':', alpha=0.7)
    ax1.set_title('Recent Data + Forecast')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Full series
    ax2.plot(data.index, data.values, alpha=0.7)
    if trend is not None:
        ax2.plot(data.index, trend.values, 'r-', alpha=0.8, label='Trend')
        ax2.legend()
    ax2.set_title('Complete Time Series')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('time_series_quick.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: time_series_quick.png")
    plt.show()

def quick_time_series_demo():
    """Complete time series analysis in 15 minutes"""
    print("🚀 QUICK TIME SERIES ANALYSIS (15 min)")
    print("=" * 40)
    
    # 1. Create data
    data = create_sample_data()
    
    # 2. Stationarity check
    is_stationary, processed = check_stationarity(data['value'])
    
    # 3. Decomposition
    trend, seasonal, residual = decompose_series(data['value'])
    
    # 4. Feature engineering
    df_features, feature_cols = create_features(data['value'])
    
    # 5. Simple forecast
    forecast = simple_forecast(data['value'])
    
    # 6. ML comparison
    compare_ml_models(df_features, feature_cols)
    
    # 7. Visualization
    create_visualization(data['value'], forecast, trend)
    
    print("\n🎯 COMPLETE! Key concepts:")
    print("  ✅ Time series = Trend + Seasonal + Noise")
    print("  ✅ Stationarity crucial for forecasting")
    print("  ✅ Feature engineering with lags & rolling stats")
    print("  ✅ Simple AR forecasting method")
    print("  ✅ ML models competitive with traditional methods")

if __name__ == "__main__":
    quick_time_series_demo()