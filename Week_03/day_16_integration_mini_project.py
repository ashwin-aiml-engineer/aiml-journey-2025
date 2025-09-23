"""
Day 16: Integration Mini-Project
🎯 Combine clustering + PCA + Docker 
✅ Real customer segmentation with containerized deployment
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
import json

# ================================
# CUSTOMER SEGMENTATION PIPELINE
# ================================

def generate_customer_data(n_customers=1000):
    """Generate realistic customer data"""
    print("🔄 Generating customer data...")
    
    np.random.seed(42)
    
    # Customer features
    data = {
        'customer_id': range(1, n_customers + 1),
        'age': np.random.normal(40, 12, n_customers).clip(18, 70),
        'annual_income': np.random.lognormal(10.8, 0.6, n_customers).clip(25000, 150000),
        'spending_score': np.random.normal(50, 25, n_customers).clip(1, 100),
        'purchase_frequency': np.random.poisson(8, n_customers),
        'avg_order_value': np.random.lognormal(4.2, 0.7, n_customers).clip(20, 500),
        'mobile_app_usage': np.random.beta(3, 5, n_customers) * 100,
        'website_visits': np.random.negative_binomial(12, 0.4, n_customers),
        'support_tickets': np.random.poisson(1.5, n_customers)
    }
    
    # Add some realistic correlations
    for i in range(n_customers):
        if data['annual_income'][i] > 75000:
            data['spending_score'][i] *= 1.2
            data['avg_order_value'][i] *= 1.3
    
    df = pd.DataFrame(data)
    print(f"✅ Generated {len(df)} customers with {len(df.columns)-1} features")
    return df

def perform_pca_clustering_pipeline(df):
    """Complete PCA + Clustering pipeline"""
    print("\n🔬 Running PCA + Clustering Pipeline...")
    
    # Prepare features (exclude customer_id)
    feature_cols = [col for col in df.columns if col != 'customer_id']
    X = df[feature_cols].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA for dimensionality reduction
    pca = PCA(n_components=5)  # Reduce to 5 dimensions
    X_pca = pca.fit_transform(X_scaled)
    
    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"  📊 PCA: Reduced to 5 components, {variance_explained:.3f} variance retained")
    
    # Find optimal clusters
    silhouette_scores = []
    k_range = range(2, 8)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        silhouette_scores.append(score)
    
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"  🎯 Optimal clusters: {optimal_k} (silhouette: {max(silhouette_scores):.3f})")
    
    # Final clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca)
    
    # Add clusters to dataframe
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    
    return df_clustered, scaler, pca, kmeans, X_pca

def analyze_customer_segments(df_clustered):
    """Analyze customer segments"""
    print("\n📊 Analyzing Customer Segments...")
    
    segments = {}
    
    for cluster_id in sorted(df_clustered['cluster'].unique()):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        
        # Calculate segment characteristics
        segment = {
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(df_clustered) * 100,
            'avg_age': cluster_data['age'].mean(),
            'avg_income': cluster_data['annual_income'].mean(),
            'avg_spending': cluster_data['spending_score'].mean(),
            'avg_frequency': cluster_data['purchase_frequency'].mean(),
            'avg_mobile_usage': cluster_data['mobile_app_usage'].mean()
        }
        
        # Segment naming based on characteristics
        if segment['avg_income'] > 75000 and segment['avg_spending'] > 65:
            name = "Premium Customers"
        elif segment['avg_frequency'] > 10 and segment['avg_mobile_usage'] > 70:
            name = "Digital Frequent Buyers"
        elif segment['avg_age'] > 50 and segment['avg_spending'] < 40:
            name = "Conservative Spenders"
        else:
            name = "Standard Customers"
        
        segment['name'] = name
        segments[cluster_id] = segment
        
        print(f"  🏷️ Cluster {cluster_id}: {name} ({segment['size']} customers, {segment['percentage']:.1f}%)")
    
    return segments

def create_visualization(df_clustered, X_pca):
    """Create customer segmentation visualization"""
    print("\n📈 Creating visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Customer Segmentation Results', fontsize=14, fontweight='bold')
    
    # PCA scatter plot
    scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=df_clustered['cluster'], 
                             cmap='viridis', alpha=0.6)
    axes[0].set_title('Customer Segments (PCA Space)')
    axes[0].set_xlabel('First Principal Component')
    axes[0].set_ylabel('Second Principal Component')
    plt.colorbar(scatter, ax=axes[0])
    
    # Cluster distribution
    cluster_counts = df_clustered['cluster'].value_counts().sort_index()
    axes[1].pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index], 
               autopct='%1.1f%%')
    axes[1].set_title('Customer Segment Distribution')
    
    plt.tight_layout()
    plt.savefig('customer_segmentation_results.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'customer_segmentation_results.png'")
    plt.show()

def save_models_and_results(scaler, pca, kmeans, segments):
    """Save models and analysis results"""
    print("\n💾 Saving models and results...")
    
    # Save models
    joblib.dump(scaler, 'customer_scaler.joblib')
    joblib.dump(pca, 'customer_pca.joblib')
    joblib.dump(kmeans, 'customer_kmeans.joblib')
    
    # Save segment analysis
    # Convert numpy types to native Python for JSON serialization
    segments_json = {}
    for k, v in segments.items():
        key = str(k)  # Convert cluster number to string
        segments_json[key] = {
            'name': v['name'],
            'size': int(v['size']),
            'percentage': float(v['percentage']),
            'avg_age': float(v['avg_age']),
            'avg_income': float(v['avg_income']),
            'avg_spending': float(v['avg_spending']),
            'avg_frequency': float(v['avg_frequency']),
            'avg_mobile_usage': float(v['avg_mobile_usage'])
        }
    
    with open('customer_segments.json', 'w') as f:
        json.dump(segments_json, f, indent=2)
    
    print("✅ Saved models and results:")
    print("  • customer_scaler.joblib")
    print("  • customer_pca.joblib")
    print("  • customer_kmeans.joblib")
    print("  • customer_segments.json")

def generate_docker_api():
    """Generate FastAPI service for customer segmentation"""
    print("\n🐳 Generating Docker deployment files...")
    
    # Simple FastAPI service
    api_code = '''"""
Customer Segmentation API
"""
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from typing import List

app = FastAPI(title="Customer Segmentation API")

# Load models on startup
scaler = joblib.load("customer_scaler.joblib")
pca = joblib.load("customer_pca.joblib")
kmeans = joblib.load("customer_kmeans.joblib")

class CustomerData(BaseModel):
    age: float
    annual_income: float
    spending_score: float
    purchase_frequency: int
    avg_order_value: float
    mobile_app_usage: float
    website_visits: int
    support_tickets: int

@app.get("/")
async def root():
    return {"message": "Customer Segmentation API Ready"}

@app.post("/predict-segment")
async def predict_segment(customer: CustomerData):
    # Prepare features
    features = np.array([[
        customer.age, customer.annual_income, customer.spending_score,
        customer.purchase_frequency, customer.avg_order_value,
        customer.mobile_app_usage, customer.website_visits, customer.support_tickets
    ]])
    
    # Transform and predict
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)
    cluster = int(kmeans.predict(features_pca)[0])
    
    return {"customer_segment": cluster}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    # Simple Dockerfile
    dockerfile = '''FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "api.py"]
'''
    
    # Requirements
    requirements = '''fastapi==0.103.1
uvicorn==0.23.2
scikit-learn==1.3.0
numpy==1.24.3
joblib==1.3.2
pydantic==2.3.0
'''
    
    # Save files
    with open('api.py', 'w') as f:
        f.write(api_code)
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile)
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("✅ Generated Docker files:")
    print("  • api.py")
    print("  • Dockerfile")
    print("  • requirements.txt")

# ================================
# MAIN INTEGRATION DEMO
# ================================

def run_integration_demo():
    """Run complete customer segmentation integration"""
    print("🚀 CUSTOMER SEGMENTATION INTEGRATION")
    print("=" * 40)
    print("🎯 PCA + Clustering + Docker in 20 minutes")
    print("=" * 40)
    
    # 1. Generate customer data
    df = generate_customer_data(n_customers=800)
    
    # 2. Run PCA + Clustering pipeline
    df_clustered, scaler, pca, kmeans, X_pca = perform_pca_clustering_pipeline(df)
    
    # 3. Analyze segments
    segments = analyze_customer_segments(df_clustered)
    
    # 4. Create visualization
    create_visualization(df_clustered, X_pca)
    
    # 5. Save models and results
    save_models_and_results(scaler, pca, kmeans, segments)
    
    # 6. Generate Docker deployment
    generate_docker_api()
    
    print("\n✅ INTEGRATION COMPLETE!")
    print("=" * 30)
    print("🎯 What you built:")
    print(f"  • Customer segmentation model ({len(segments)} segments)")
    print("  • PCA dimensionality reduction pipeline")
    print("  • Containerized prediction API")
    print("  • Complete deployment setup")
    
    print("\n🐳 Docker Commands:")
    print("  1. docker build -t customer-segmentation .")
    print("  2. docker run -p 8000:8000 customer-segmentation")
    print("  3. Test: curl http://localhost:8000/")
    
    print("\n💡 Business Value:")
    print("  ✅ Automated customer segmentation")
    print("  ✅ Real-time segment prediction")
    print("  ✅ Scalable containerized deployment")
    print("  ✅ Data-driven marketing insights")

if __name__ == "__main__":
    run_integration_demo()