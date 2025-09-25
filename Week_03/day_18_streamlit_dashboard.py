"""
Day 18: Streamlit Dashboard (15-Minute Daily Practice)
🎯 Build interactive ML web apps quickly
✅ Widgets, visualization, file upload, model training
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main Streamlit app"""
    st.set_page_config(page_title="ML Dashboard", layout="wide")
    
    st.title("🚀 Quick ML Dashboard")
    st.markdown("**15-minute Streamlit practice - Interactive ML training & visualization**")
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Data options
        data_source = st.selectbox("Data Source", ["Generate Sample", "Upload File"])
        
        if data_source == "Generate Sample":
            n_samples = st.slider("Sample Size", 100, 1000, 500)
            noise_level = st.slider("Noise Level", 0.1, 2.0, 0.5)
        
        # Model options
        st.subheader("🤖 Model Settings")
        model_type = st.selectbox("Model", ["Random Forest", "Linear Regression"])
        
        if model_type == "Random Forest":
            n_trees = st.slider("Trees", 10, 200, 50)
            max_depth = st.slider("Max Depth", 3, 20, 10)
        
        train_button = st.button("🔥 Train Model", type="primary")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Load/generate data
        data = load_data(data_source, n_samples if data_source == "Generate Sample" else None,
                        noise_level if data_source == "Generate Sample" else None)
        
        if data is not None:
            st.subheader("📊 Data Preview")
            st.dataframe(data.head(), use_container_width=True)
            
            # Interactive plot
            fig = px.line(data, y='value', title='Time Series Data')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Stats")
        if data is not None:
            st.metric("Data Points", len(data))
            st.metric("Mean", f"{data['value'].mean():.2f}")
            st.metric("Std Dev", f"{data['value'].std():.2f}")
        
        # Model training
        if train_button and data is not None:
            with st.spinner("Training model..."):
                results = train_model(data, model_type, 
                                    {'n_trees': n_trees, 'max_depth': max_depth} if model_type == "Random Forest" else {})
            
            st.success("✅ Model trained!")
            st.metric("MSE", f"{results['mse']:.3f}")
            st.metric("R²", f"{results['r2']:.3f}")
    
    # Additional features
    st.subheader("🔮 Quick Forecast")
    if data is not None:
        forecast_days = st.slider("Forecast Days", 7, 60, 30)
        
        if st.button("Generate Forecast"):
            forecast = simple_forecast(data, forecast_days)
            
            # Combined plot
            fig = px.line(title=f'{forecast_days}-Day Forecast')
            
            # Recent actual data
            recent = data.tail(100)
            fig.add_scatter(x=recent.index, y=recent['value'], name='Actual', mode='lines')
            
            # Forecast
            fig.add_scatter(x=forecast.index, y=forecast['forecast'], 
                          name='Forecast', mode='lines', line_dash='dash')
            
            st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def load_data(source, n_samples=None, noise=None):
    """Load or generate data"""
    if source == "Generate Sample":
        # Generate sample time series
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
        trend = np.linspace(100, 150, n_samples)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
        noise_component = np.random.normal(0, noise, n_samples)
        
        values = trend + seasonal + noise_component
        return pd.DataFrame({'value': values}, index=dates)
    
    return None

def train_model(data, model_type, params):
    """Train ML model"""
    # Create features
    df = data.copy()
    for lag in [1, 2, 3, 7]:
        df[f'lag_{lag}'] = df['value'].shift(lag)
    
    df['ma_7'] = df['value'].rolling(7).mean()
    df['day_of_week'] = df.index.dayofweek
    
    # Clean data
    df_clean = df.dropna()
    
    # Features and target
    feature_cols = [col for col in df_clean.columns if col != 'value']
    X = df_clean[feature_cols]
    y = df_clean['value']
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train model
    if model_type == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=params['n_trees'],
            max_depth=params['max_depth'],
            random_state=42
        )
    else:
        model = LinearRegression()
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = model.score(X_test, y_test)
    
    return {'mse': mse, 'r2': r2, 'model': model}

def simple_forecast(data, periods):
    """Simple forecast using last values"""
    # Simple moving average forecast
    window = min(30, len(data) // 4)
    recent_mean = data['value'].tail(window).mean()
    recent_trend = (data['value'].iloc[-1] - data['value'].iloc[-window]) / window
    
    # Generate forecast
    forecast_values = []
    for i in range(periods):
        next_val = recent_mean + recent_trend * i
        forecast_values.append(next_val)
    
    forecast_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=periods)
    return pd.DataFrame({'forecast': forecast_values}, index=forecast_dates)

# File upload widget
def file_upload_section():
    """File upload functionality"""
    st.subheader("📁 Upload Your Data")
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=True, index_col=0)
            st.success(f"✅ Loaded {len(df)} rows")
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    return None

if __name__ == "__main__":
    # Add file upload option
    with st.expander("📁 Advanced: Upload Your Own Data"):
        uploaded_data = file_upload_section()
        if uploaded_data is not None:
            st.write("Uploaded data preview:")
            st.dataframe(uploaded_data.head())
    
    main()
    
    # Footer
    st.markdown("---")
    st.markdown("**🎯 15-Min Streamlit Practice Complete!** Built interactive dashboard with widgets, ML training, and forecasting.")