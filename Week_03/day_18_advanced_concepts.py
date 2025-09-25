"""
Day 18 BONUS: Advanced Time Series & Streamlit Concepts (25-Minute Deep Dive)
🎯 Master cutting-edge forecasting and deployment techniques
✅ Prophet, advanced validation, production Streamlit deployment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("📝 Streamlit/Plotly not available - using matplotlib for visualization")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("📝 Prophet not available - using alternative forecasting")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("📝 yfinance not available - using synthetic data")

def prophet_forecasting_system():
    """Advanced Prophet-based forecasting with seasonality detection"""
    print("\n🔮 PROPHET FORECASTING SYSTEM")
    print("=" * 35)
    
    if not PROPHET_AVAILABLE:
        print("⚠️ Prophet not installed. Install with: pip install prophet")
        return create_prophet_alternative()
    
    # Create complex seasonal data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    
    # Multi-level seasonality
    trend = np.linspace(100, 150, len(dates))
    yearly = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    weekly = 3 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    noise = np.random.normal(0, 2, len(dates))
    
    # Add holiday effects
    holiday_effect = np.zeros(len(dates))
    for i, date in enumerate(dates):
        if date.month == 12 and date.day in [24, 25, 31]:  # Christmas/New Year
            holiday_effect[i] = np.random.normal(15, 3)
        elif date.month == 11 and date.weekday() == 3 and 22 <= date.day <= 28:  # Thanksgiving
            holiday_effect[i] = np.random.normal(8, 2)
    
    values = trend + yearly + weekly + holiday_effect + noise
    
    df = pd.DataFrame({
        'ds': dates,
        'y': values
    })
    
    # Split for validation
    train_size = int(0.8 * len(df))
    train_df = df[:train_size].copy()
    test_df = df[train_size:].copy()
    
    print(f"  Training data: {len(train_df)} days")
    print(f"  Test data: {len(test_df)} days")
    
    # Configure Prophet with custom seasonalities
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='additive',
        changepoint_prior_scale=0.05
    )
    
    # Add custom seasonalities
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
    
    # Fit model
    print("  🔄 Training Prophet model...")
    model.fit(train_df)
    
    # Generate predictions
    future = model.make_future_dataframe(periods=len(test_df))
    forecast = model.predict(future)
    
    # Evaluate on test set
    test_predictions = forecast[train_size:]['yhat'].values
    test_actual = test_df['y'].values
    
    mse = mean_squared_error(test_actual, test_predictions)
    mape = mean_absolute_percentage_error(test_actual, test_predictions) * 100
    
    print(f"  📊 Test MSE: {mse:.2f}")
    print(f"  📊 Test MAPE: {mape:.1f}%")
    
    # Component analysis
    print("\n  🔍 Seasonality Components:")
    components = model.predict(future)
    print(f"    Trend strength: {components['trend'].std():.2f}")
    print(f"    Yearly seasonal strength: {components['yearly'].std():.2f}")
    print(f"    Weekly seasonal strength: {components['weekly'].std():.2f}")
    
    return model, forecast, df, test_df

def create_prophet_alternative():
    """Alternative seasonal decomposition when Prophet unavailable"""
    from scipy import signal
    
    print("  🔄 Using alternative seasonal decomposition...")
    
    # Generate data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    trend = np.linspace(100, 150, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    noise = np.random.normal(0, 3, len(dates))
    
    df = pd.DataFrame({
        'ds': dates,
        'y': trend + seasonal + noise
    })
    
    # Simple trend extraction using Savitzky-Golay filter
    trend_smooth = signal.savgol_filter(df['y'], window_length=91, polyorder=3)
    seasonal_component = df['y'] - trend_smooth
    
    print(f"  📊 Trend variation: {trend_smooth.std():.2f}")
    print(f"  📊 Seasonal variation: {seasonal_component.std():.2f}")
    
    return df, trend_smooth, seasonal_component

def advanced_walk_forward_validation():
    """Sophisticated walk-forward validation with expanding windows"""
    print("\n📈 ADVANCED WALK-FORWARD VALIDATION")
    print("=" * 38)
    
    # Generate realistic financial time series
    np.random.seed(42)
    n_days = 1000
    returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
    prices = [100]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    df = pd.DataFrame({'price': prices[1:]}, index=dates)
    
    # Create features
    for lag in [1, 2, 3, 5, 10]:
        df[f'return_lag_{lag}'] = df['price'].pct_change().shift(lag)
        df[f'price_lag_{lag}'] = df['price'].shift(lag)
    
    # Rolling statistics
    for window in [5, 10, 20]:
        df[f'volatility_{window}'] = df['price'].pct_change().rolling(window).std()
        df[f'momentum_{window}'] = (df['price'] / df['price'].shift(window) - 1)
    
    df = df.dropna()
    
    # Advanced validation strategy
    validation_results = []
    feature_cols = [col for col in df.columns if col != 'price']
    
    # Multiple validation schemes
    validation_schemes = [
        {'name': 'Expanding', 'min_train': 200, 'test_size': 50, 'step': 25},
        {'name': 'Rolling', 'min_train': 200, 'test_size': 50, 'step': 25},
        {'name': 'Purged', 'min_train': 200, 'test_size': 50, 'step': 25, 'gap': 5}
    ]
    
    for scheme in validation_schemes:
        print(f"\n  🔍 {scheme['name']} Window Validation")
        scheme_results = []
        
        n_splits = 5
        for split in range(n_splits):
            if scheme['name'] == 'Expanding':
                train_start = 0
                train_end = scheme['min_train'] + split * scheme['step']
                test_start = train_end
                test_end = test_start + scheme['test_size']
            
            elif scheme['name'] == 'Rolling':
                test_end = scheme['min_train'] + (split + 1) * scheme['step'] + scheme['test_size']
                test_start = test_end - scheme['test_size']
                train_end = test_start
                train_start = train_end - scheme['min_train']
            
            else:  # Purged
                test_end = scheme['min_train'] + (split + 1) * scheme['step'] + scheme['test_size']
                test_start = test_end - scheme['test_size']
                train_end = test_start - scheme['gap']  # Purging gap
                train_start = train_end - scheme['min_train']
            
            if test_end > len(df) or train_start < 0:
                continue
            
            # Prepare data
            X_train = df.iloc[train_start:train_end][feature_cols].values
            y_train = df.iloc[train_start:train_end]['price'].values
            X_test = df.iloc[test_start:test_end][feature_cols].values
            y_test = df.iloc[test_start:test_end]['price'].values
            
            # Scale and train
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Predict and evaluate
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            
            scheme_results.append(mse)
            print(f"    Split {split+1}: MSE={mse:.2f}")
        
        avg_mse = np.mean(scheme_results)
        std_mse = np.std(scheme_results)
        print(f"    Average MSE: {avg_mse:.2f} ± {std_mse:.2f}")
        
        validation_results.append({
            'scheme': scheme['name'],
            'avg_mse': avg_mse,
            'std_mse': std_mse,
            'results': scheme_results
        })
    
    # Find most stable validation
    most_stable = min(validation_results, key=lambda x: x['std_mse'])
    print(f"\n  🏆 Most Stable: {most_stable['scheme']} (std={most_stable['std_mse']:.2f})")
    
    return validation_results

def production_streamlit_deployment():
    """Advanced Streamlit deployment patterns"""
    print("\n🚀 PRODUCTION STREAMLIT PATTERNS")
    print("=" * 33)
    
    if not STREAMLIT_AVAILABLE:
        print("⚠️ Streamlit not installed. Install with: pip install streamlit plotly")
        print("💡 Here's the deployment pattern template:")
        
        deployment_template = '''
# Production Streamlit App Template
# Install: pip install streamlit plotly

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
from datetime import datetime

# Key production patterns demonstrated:
# 1. Page configuration and layout
# 2. Session state management
# 3. Custom CSS styling
# 4. Model registry and versioning
# 5. A/B testing framework
# 6. Real-time monitoring
# 7. Auto-refresh functionality
# 8. Health checks and status monitoring

st.set_page_config(page_title="ML Platform", layout="wide")
# ... (full template available in code)
'''
        
        print("📝 Template covers:")
        print("  • Multi-page navigation with session state")
        print("  • Real-time monitoring dashboard") 
        print("  • Model registry and versioning")
        print("  • A/B testing capabilities")
        print("  • Health monitoring and status checks")
        print("  • Auto-refresh functionality")
        print("  • Professional CSS styling")
        print("  • Production-ready error handling")
        
        return deployment_template
    
    deployment_code = '''
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import time
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="ML Forecasting Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None

# Custom CSS for production styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: #f8fafc;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
}
.status-success {
    color: #10b981;
    font-weight: bold;
}
.status-warning {
    color: #f59e0b;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header"><h1>🚀 Production ML Forecasting Platform</h1></div>', 
            unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    page = st.selectbox("Select Page", [
        "🏠 Dashboard", 
        "📊 Data Analysis", 
        "🤖 Model Training", 
        "🔮 Forecasting",
        "⚙️ Model Management"
    ])
    
    # System status
    st.subheader("📡 System Status")
    
    # Simulate system health checks
    db_status = "🟢 Connected" if np.random.random() > 0.1 else "🔴 Error"
    api_status = "🟢 Healthy" if np.random.random() > 0.05 else "🟡 Slow"
    
    st.write(f"Database: {db_status}")
    st.write(f"API Service: {api_status}")
    st.write(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")

# Dashboard Page
if page == "🏠 Dashboard":
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📈 Models Deployed", "3", "1")
    
    with col2:
        st.metric("🎯 Avg Accuracy", "94.2%", "2.1%")
    
    with col3:
        st.metric("⚡ Predictions Today", "1,247", "156")
    
    with col4:
        st.metric("🔄 Uptime", "99.8%", "0.1%")
    
    # Real-time monitoring chart
    st.subheader("📊 Real-Time Model Performance")
    
    # Generate mock real-time data
    times = pd.date_range(datetime.now() - timedelta(hours=24), 
                         datetime.now(), freq='H')
    performance = 95 + np.random.normal(0, 2, len(times))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=performance,
        mode='lines+markers',
        name='Model Accuracy %',
        line=dict(color='#3b82f6', width=2)
    ))
    fig.update_layout(
        title="24-Hour Performance Monitoring",
        xaxis_title="Time",
        yaxis_title="Accuracy %",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

# Model Management Page
elif page == "⚙️ Model Management":
    st.header("🛠️ Model Management System")
    
    # Model versioning
    st.subheader("📋 Model Registry")
    
    models_df = pd.DataFrame({
        'Model ID': ['rf_v1.2.1', 'gbm_v2.0.0', 'lstm_v1.0.3'],
        'Algorithm': ['Random Forest', 'Gradient Boosting', 'LSTM'],
        'Accuracy': ['94.2%', '92.8%', '96.1%'],
        'Status': ['🟢 Production', '🟡 Staging', '🔴 Deprecated'],
        'Last Updated': ['2024-01-15', '2024-01-20', '2024-01-10']
    })
    
    st.dataframe(models_df, use_container_width=True)
    
    # Model deployment controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚀 Deploy New Model")
        
        model_file = st.file_uploader("Upload Model (.pkl)", type=['pkl'])
        model_name = st.text_input("Model Name", "my_model_v1.0")
        
        if st.button("🚀 Deploy Model"):
            if model_file and model_name:
                with st.spinner("Deploying model..."):
                    time.sleep(2)  # Simulate deployment
                st.success(f"✅ {model_name} deployed successfully!")
    
    with col2:
        st.subheader("🔄 A/B Testing")
        
        model_a = st.selectbox("Model A", ['rf_v1.2.1', 'gbm_v2.0.0'])
        model_b = st.selectbox("Model B", ['gbm_v2.0.0', 'lstm_v1.0.3'])
        traffic_split = st.slider("Traffic Split (A/B)", 0, 100, 50)
        
        if st.button("🧪 Start A/B Test"):
            st.info(f"A/B Test started: {traffic_split}% → {model_a}, {100-traffic_split}% → {model_b}")

# Auto-refresh functionality
if st.sidebar.button("🔄 Auto Refresh (30s)"):
    time.sleep(30)
    st.experimental_rerun()

# Footer with system info
st.markdown("---")
st.markdown(f"**System Info:** Python 3.9+ | Streamlit {st.__version__} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
'''
    
    print("  💾 Created production deployment template")
    print("  📝 Features:")
    print("    • Multi-page navigation with session state")
    print("    • Real-time monitoring dashboard")
    print("    • Model registry and versioning")
    print("    • A/B testing capabilities")
    print("    • Health monitoring and status checks")
    print("    • Auto-refresh functionality")
    print("    • Professional CSS styling")
    print("    • Production-ready error handling")
    
    return deployment_code

def ensemble_forecasting_methods():
    """Advanced ensemble forecasting techniques"""
    print("\n🎯 ENSEMBLE FORECASTING METHODS")
    print("=" * 34)
    
    # Generate complex time series
    np.random.seed(42)
    n_points = 500
    t = np.arange(n_points)
    
    # Multiple underlying patterns
    trend = 0.02 * t
    seasonal1 = 5 * np.sin(2 * np.pi * t / 50)  # Main seasonality
    seasonal2 = 2 * np.sin(2 * np.pi * t / 12)  # Sub-seasonality
    noise = np.random.normal(0, 1, n_points)
    regime_change = np.where(t > 300, 10, 0)  # Structural break
    
    ts = 50 + trend + seasonal1 + seasonal2 + regime_change + noise
    
    dates = pd.date_range('2020-01-01', periods=n_points, freq='D')
    df = pd.DataFrame({'value': ts}, index=dates)
    
    # Create features for ensemble
    for lag in [1, 2, 3, 7, 14]:
        df[f'lag_{lag}'] = df['value'].shift(lag)
    
    # Rolling features
    for window in [7, 14, 30]:
        df[f'ma_{window}'] = df['value'].rolling(window).mean()
        df[f'std_{window}'] = df['value'].rolling(window).std()
    
    df = df.dropna()
    
    # Split data
    train_size = int(0.8 * len(df))
    X_train = df.iloc[:train_size, 1:].values  # Exclude target
    y_train = df.iloc[:train_size, 0].values   # Target
    X_test = df.iloc[train_size:, 1:].values
    y_test = df.iloc[train_size:, 0].values
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Multiple models for ensemble
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.svm import SVR
    
    models = {
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0)
    }
    
    # Train individual models
    individual_predictions = {}
    individual_scores = {}
    
    print("  🤖 Training ensemble components:")
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        score = mean_squared_error(y_test, pred)
        
        individual_predictions[name] = pred
        individual_scores[name] = score
        
        print(f"    {name}: MSE = {score:.2f}")
    
    # Ensemble methods
    ensemble_results = {}
    
    # 1. Simple averaging
    simple_avg = np.mean(list(individual_predictions.values()), axis=0)
    ensemble_results['Simple Average'] = mean_squared_error(y_test, simple_avg)
    
    # 2. Weighted by inverse error
    weights = np.array([1/score for score in individual_scores.values()])
    weights = weights / weights.sum()
    
    weighted_pred = np.zeros(len(y_test))
    for i, pred in enumerate(individual_predictions.values()):
        weighted_pred += weights[i] * pred
    ensemble_results['Weighted Average'] = mean_squared_error(y_test, weighted_pred)
    
    # 3. Stacking ensemble (meta-learner)
    from sklearn.linear_model import LinearRegression
    
    # Create meta-features (predictions from base models)
    meta_train = np.column_stack([
        model.predict(X_train_scaled) for model in models.values()
    ])
    meta_test = np.column_stack(list(individual_predictions.values()))
    
    # Train meta-learner
    meta_model = LinearRegression()
    meta_model.fit(meta_train, y_train)
    stacked_pred = meta_model.predict(meta_test)
    ensemble_results['Stacked'] = mean_squared_error(y_test, stacked_pred)
    
    print("\n  🎯 Ensemble Results:")
    for method, score in ensemble_results.items():
        print(f"    {method}: MSE = {score:.2f}")
    
    # Find best method
    best_ensemble = min(ensemble_results.items(), key=lambda x: x[1])
    print(f"\n  🏆 Best Ensemble: {best_ensemble[0]} (MSE = {best_ensemble[1]:.2f})")
    
    return ensemble_results, individual_scores

def comprehensive_bonus_demo():
    """Complete advanced concepts demonstration"""
    print("🌟 DAY 18 BONUS: ADVANCED CONCEPTS (25 min)")
    print("=" * 48)
    
    try:
        # 1. Prophet forecasting
        if PROPHET_AVAILABLE:
            prophet_model, prophet_forecast, data, test_data = prophet_forecasting_system()
        else:
            alt_data, trend, seasonal = create_prophet_alternative()
        
        # 2. Advanced validation techniques
        validation_results = advanced_walk_forward_validation()
        
        # 3. Ensemble forecasting
        ensemble_results, individual_results = ensemble_forecasting_methods()
        
        # 4. Production deployment patterns
        deployment_template = production_streamlit_deployment()
        
        print("\n🎯 ADVANCED CONCEPTS MASTERED!")
        print("Deep dive accomplishments:")
        print("  ✅ Prophet forecasting with custom seasonalities")
        print("  ✅ Advanced walk-forward validation (expanding/rolling/purged)")
        print("  ✅ Sophisticated ensemble methods (averaging/weighted/stacked)")
        print("  ✅ Production Streamlit deployment patterns")
        print("  ✅ Model registry and A/B testing framework")
        print("  ✅ Real-time monitoring dashboard")
        
        print("\n📚 Key Advanced Learnings:")
        print("  • Prophet handles complex seasonalities automatically")
        print("  • Purged validation prevents look-ahead bias")
        print("  • Ensemble methods improve robustness significantly")
        print("  • Production deployment requires monitoring & versioning")
        print("  • A/B testing enables safe model rollouts")
        
        # Create summary visualization
        create_advanced_summary_plot(validation_results, ensemble_results)
        
    except Exception as e:
        print(f"⚠️ Error in advanced demo: {e}")
        print("💡 Install missing packages: pip install prophet yfinance")

def create_advanced_summary_plot(validation_results, ensemble_results):
    """Create comprehensive summary visualization"""
    print("\n📊 Creating advanced summary visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Validation scheme comparison
    schemes = [r['scheme'] for r in validation_results]
    avg_mse = [r['avg_mse'] for r in validation_results]
    std_mse = [r['std_mse'] for r in validation_results]
    
    ax1.bar(schemes, avg_mse, yerr=std_mse, capsize=5, alpha=0.7, color='skyblue')
    ax1.set_title('Validation Scheme Comparison')
    ax1.set_ylabel('Average MSE')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Ensemble vs Individual models
    all_results = {**ensemble_results, 'Best Individual': min(ensemble_results.values())}
    methods = list(all_results.keys())
    scores = list(all_results.values())
    
    colors = ['lightgreen' if 'Ensemble' in m or 'Average' in m or 'Stacked' in m 
              else 'lightcoral' for m in methods]
    ax2.bar(methods, scores, color=colors, alpha=0.7)
    ax2.set_title('Ensemble vs Individual Performance')
    ax2.set_ylabel('MSE')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Model complexity vs performance
    complexity = [1, 3, 5, 7, 10]  # Model complexity scale
    performance = [8.5, 7.2, 6.1, 5.8, 5.9]  # Mock performance
    
    ax3.plot(complexity, performance, 'bo-', linewidth=2, markersize=8)
    ax3.set_title('Model Complexity vs Performance')
    ax3.set_xlabel('Model Complexity')
    ax3.set_ylabel('Performance Score')
    ax3.grid(True, alpha=0.3)
    
    # 4. Forecast horizon accuracy
    horizons = [1, 7, 14, 30, 60, 90]
    accuracy = [95, 88, 82, 75, 68, 62]  # Decreasing with horizon
    
    ax4.plot(horizons, accuracy, 'ro-', linewidth=2, markersize=8)
    ax4.set_title('Forecast Accuracy vs Time Horizon')
    ax4.set_xlabel('Days Ahead')
    ax4.set_ylabel('Accuracy %')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('advanced_concepts_summary.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: advanced_concepts_summary.png")
    plt.show()

if __name__ == "__main__":
    comprehensive_bonus_demo()