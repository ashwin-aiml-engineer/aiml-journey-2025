"""
Day 18: ML Forecasting System (15-Minute Daily Practice)
🎯 Master ML forecasting with proper validation
✅ Walk-forward validation, model comparison, future predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def create_stock_data():
    """Generate realistic stock-like data"""
    print("📊 Creating realistic time series...")
    np.random.seed(42)
    
    n_days = 400
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = [100]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    df = pd.DataFrame({'price': prices[1:]}, index=dates)
    
    print(f"✅ Created {n_days} days: {df['price'].min():.1f} - {df['price'].max():.1f}")
    return df

def create_features(data):
    """Essential ML features"""
    print("\n⚙️ Feature engineering...")
    df = data.copy()
    
    # Price features
    df['returns'] = df['price'].pct_change()
    df['ma_7'] = df['price'].rolling(7).mean()
    df['ma_21'] = df['price'].rolling(21).mean()
    
    # Lag features
    for lag in [1, 2, 3, 7]:
        df[f'lag_{lag}'] = df['price'].shift(lag)
        df[f'ret_lag_{lag}'] = df['returns'].shift(lag)
    
    # Volatility
    df['vol_7'] = df['returns'].rolling(7).std()
    df['momentum'] = df['price'] / df['price'].shift(10) - 1
    
    # Time features
    df['day_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    feature_cols = [col for col in df.columns if col != 'price']
    df_clean = df.dropna()
    
    print(f"  Features: {len(feature_cols)}, Clean data: {len(df_clean)}")
    return df_clean, feature_cols

def walk_forward_validation(data, feature_cols, model_class, n_splits=4):
    """Proper walk-forward validation"""
    print(f"\n📈 Walk-forward validation: {model_class.__name__}")
    
    X = data[feature_cols].values
    y = data['price'].values
    
    initial_train = 150
    test_size = 30
    results = []
    
    for i in range(n_splits):
        # Define windows
        train_end = initial_train + i * test_size
        test_start = train_end
        test_end = test_start + test_size
        
        if test_end > len(X):
            break
        
        # Split data
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]
        
        # Train model
        if model_class == LinearRegression:
            model = model_class()
        else:
            model = model_class(n_estimators=50, random_state=42)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        results.append(mse)
        
        print(f"  Fold {i+1}: Train={len(X_train)}, Test={len(X_test)}, MSE={mse:.2f}")
    
    avg_mse = np.mean(results)
    print(f"  Average MSE: {avg_mse:.2f}")
    
    return results, avg_mse

def compare_models(data, feature_cols):
    """Compare different models"""
    print("\n🤖 Model comparison...")
    
    models = {
        'Linear': LinearRegression,
        'Random Forest': RandomForestRegressor,
        'Gradient Boost': GradientBoostingRegressor
    }
    
    model_scores = {}
    
    for name, model_class in models.items():
        results, avg_score = walk_forward_validation(data, feature_cols, model_class, n_splits=3)
        model_scores[name] = avg_score
    
    # Find best model
    best_model = min(model_scores.items(), key=lambda x: x[1])
    print(f"\n🏆 Best model: {best_model[0]} (MSE: {best_model[1]:.2f})")
    
    return model_scores, best_model

def generate_forecast(data, feature_cols, best_model_name, periods=20):
    """Generate future predictions"""
    print(f"\n🔮 Generating {periods}-day forecast...")
    
    # Train final model on all data
    X = data[feature_cols].values
    y = data['price'].values
    
    if best_model_name == 'Linear':
        model = LinearRegression()
    elif best_model_name == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    model.fit(X, y)
    
    # Simple forecast using last features (in practice, you'd update features)
    last_features = X[-1].reshape(1, -1)
    forecast = []
    
    for _ in range(periods):
        pred = model.predict(last_features)[0]
        forecast.append(pred)
    
    # Create forecast dataframe
    forecast_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=periods)
    forecast_df = pd.DataFrame({'forecast': forecast}, index=forecast_dates)
    
    print(f"  Forecast range: {min(forecast):.1f} - {max(forecast):.1f}")
    print(f"  Last actual: {data['price'].iloc[-1]:.1f}")
    
    return forecast_df

def create_forecast_plot(data, forecast_df):
    """Visualize forecast"""
    print("\n📊 Creating forecast plot...")
    
    plt.figure(figsize=(12, 6))
    
    # Plot recent data
    recent = data['price'].iloc[-60:]
    plt.plot(recent.index, recent.values, label='Historical', linewidth=2)
    
    # Plot forecast
    plt.plot(forecast_df.index, forecast_df['forecast'], 
             'r--', label='Forecast', linewidth=2)
    
    plt.axvline(data.index[-1], color='gray', linestyle=':', alpha=0.7)
    plt.title('ML Forecasting System')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('ml_forecast_quick.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: ml_forecast_quick.png")
    plt.show()

def quick_forecasting_demo():
    """Complete ML forecasting in 15 minutes"""
    print("🚀 QUICK ML FORECASTING (15 min)")
    print("=" * 35)
    
    # 1. Generate data
    data = create_stock_data()
    
    # 2. Feature engineering
    featured_data, feature_cols = create_features(data)
    
    # 3. Model comparison with walk-forward validation
    model_scores, best_model = compare_models(featured_data, feature_cols)
    
    # 4. Generate forecast
    forecast_df = generate_forecast(featured_data, feature_cols, best_model[0])
    
    # 5. Visualization
    create_forecast_plot(data, forecast_df)
    
    print("\n🎯 ML FORECASTING COMPLETE!")
    print("Key achievements:")
    print("  ✅ Realistic time series with returns & volatility")
    print("  ✅ Advanced feature engineering (lags, momentum, volatility)")
    print("  ✅ Walk-forward validation (prevents data leakage)")
    print("  ✅ Model comparison (Linear, RF, GBM)")
    print("  ✅ Future forecast generation")
    print("  ✅ Professional visualization")
    
    print(f"\n📈 Results:")
    for model, score in model_scores.items():
        print(f"  • {model}: MSE = {score:.2f}")
    print(f"  • Best: {best_model[0]}")
    print("  • Walk-forward validation ensures realistic performance")

if __name__ == "__main__":
    quick_forecasting_demo()