"""
Day 16: Clustering Analysis (Concise Daily Practice)
🎯 Master K-Means, Hierarchical, DBSCAN, GMM
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs, make_circles
import warnings
warnings.filterwarnings('ignore')

# ================================
# ESSENTIAL CLUSTERING ALGORITHMS
# ================================

def create_sample_datasets():
    """Create sample datasets for clustering analysis"""
    print("🔄 Creating clustering datasets...")
    
    # Dataset 1: Blob clusters (ideal for K-means)
    X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)
    
    # Dataset 2: Circular clusters (challenging for K-means)
    X_circles, y_circles = make_circles(n_samples=300, noise=0.1, factor=0.6, random_state=42)
    
    # Dataset 3: Customer data (realistic business case)
    np.random.seed(42)
    n_customers = 300
    X_customers = np.column_stack([
        np.random.normal(35, 10, n_customers),  # age
        np.random.normal(50000, 20000, n_customers),  # income
        np.random.normal(50, 25, n_customers),  # spending_score
        np.random.poisson(5, n_customers)  # purchase_frequency
    ])
    
    datasets = {
        'blobs': (X_blobs, "Blob Clusters"),
        'circles': (X_circles, "Circular Clusters"), 
        'customers': (X_customers, "Customer Data")
    }
    
    print(f"✅ Created {len(datasets)} datasets for clustering")
    return datasets

def find_optimal_clusters(X, max_k=8):
    """Find optimal number of clusters using elbow method"""
    print("🔍 Finding optimal clusters...")
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))
    
    # Find best silhouette score
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"  🎯 Optimal clusters: {optimal_k} (silhouette: {max(silhouette_scores):.3f})")
    
    return optimal_k, k_range, inertias, silhouette_scores

def perform_kmeans_clustering(X, n_clusters=4):
    """K-Means clustering analysis"""
    print(f"🎯 K-Means Clustering (k={n_clusters})")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Metrics
    silhouette_avg = silhouette_score(X, labels)
    inertia = kmeans.inertia_
    
    print(f"  📊 Silhouette: {silhouette_avg:.3f}, Inertia: {inertia:.2f}")
    
    return labels, {'silhouette': silhouette_avg, 'inertia': inertia}

def perform_hierarchical_clustering(X, n_clusters=4):
    """Hierarchical clustering analysis"""
    print(f"🌳 Hierarchical Clustering (k={n_clusters})")
    
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = hierarchical.fit_predict(X)
    
    # Metrics
    silhouette_avg = silhouette_score(X, labels)
    
    print(f"  📊 Silhouette: {silhouette_avg:.3f}")
    
    return labels, {'silhouette': silhouette_avg}

def perform_dbscan_clustering(X):
    """DBSCAN clustering analysis"""
    print("🎯 DBSCAN Clustering")
    
    # Auto-tune eps based on data scale
    from sklearn.neighbors import NearestNeighbors
    neighbors = NearestNeighbors(n_neighbors=4)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    distances = np.sort(distances[:, 3], axis=0)
    eps = np.percentile(distances, 90)  # Use 90th percentile
    
    dbscan = DBSCAN(eps=eps, min_samples=4)
    labels = dbscan.fit_predict(X)
    
    # Metrics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    if n_clusters > 1:
        silhouette_avg = silhouette_score(X, labels)
        print(f"  📊 Clusters: {n_clusters}, Noise: {n_noise}, Silhouette: {silhouette_avg:.3f}")
    else:
        silhouette_avg = -1
        print(f"  📊 Clusters: {n_clusters}, Noise: {n_noise} (No valid clustering)")
    
    return labels, {'silhouette': silhouette_avg, 'n_clusters': n_clusters, 'n_noise': n_noise}

def perform_gmm_clustering(X, n_components=4):
    """Gaussian Mixture Model clustering"""
    print(f"📊 GMM Clustering (k={n_components})")
    
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    labels = gmm.fit_predict(X)
    
    # Metrics
    silhouette_avg = silhouette_score(X, labels)
    aic = gmm.aic(X)
    bic = gmm.bic(X)
    
    print(f"  📊 Silhouette: {silhouette_avg:.3f}, AIC: {aic:.2f}, BIC: {bic:.2f}")
    
    return labels, {'silhouette': silhouette_avg, 'aic': aic, 'bic': bic}

def compare_clustering_algorithms(X, dataset_name, n_clusters=4):
    """Compare all clustering algorithms"""
    print(f"\n🔬 CLUSTERING COMPARISON: {dataset_name}")
    print("=" * 50)
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run all algorithms
    results = {}
    
    # K-Means
    labels_km, metrics_km = perform_kmeans_clustering(X_scaled, n_clusters)
    results['K-Means'] = (labels_km, metrics_km)
    
    # Hierarchical
    labels_hc, metrics_hc = perform_hierarchical_clustering(X_scaled, n_clusters)
    results['Hierarchical'] = (labels_hc, metrics_hc)
    
    # DBSCAN
    labels_db, metrics_db = perform_dbscan_clustering(X_scaled)
    results['DBSCAN'] = (labels_db, metrics_db)
    
    # GMM
    labels_gmm, metrics_gmm = perform_gmm_clustering(X_scaled, n_clusters)
    results['GMM'] = (labels_gmm, metrics_gmm)
    
    return results, X_scaled

def create_clustering_visualization(results, X_scaled, dataset_name):
    """Create visualization comparing all clustering methods"""
    print(f"\n📊 Creating visualization for {dataset_name}...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Clustering Comparison: {dataset_name}', fontsize=14, fontweight='bold')
    
    algorithms = ['K-Means', 'Hierarchical', 'DBSCAN', 'GMM']
    positions = [(0,0), (0,1), (1,0), (1,1)]
    
    for i, (alg, pos) in enumerate(zip(algorithms, positions)):
        if alg in results:
            labels, metrics = results[alg]
            
            # Create scatter plot
            scatter = axes[pos].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, 
                                      cmap='viridis', alpha=0.6, s=30)
            axes[pos].set_title(f'{alg}\nSilhouette: {metrics["silhouette"]:.3f}')
            axes[pos].set_xlabel('Feature 1')
            axes[pos].set_ylabel('Feature 2')
    
    plt.tight_layout()
    filename = f'clustering_comparison_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization: {filename}")
    plt.show()

def show_clustering_insights():
    """Show insights about clustering algorithms"""
    insights = """
💡 CLUSTERING ALGORITHMS COMPARISON
===================================

🔹 K-Means:
  ✅ Best for: Spherical, similar-sized clusters
  ⚠️ Assumes: Known number of clusters, spherical shape
  🎯 Use when: Clear cluster expectations, fast processing needed

🔹 Hierarchical:
  ✅ Best for: Nested clusters, unknown cluster count
  ⚠️ Assumes: Distance-based similarity
  🎯 Use when: Need cluster hierarchy, small-medium datasets

🔹 DBSCAN:
  ✅ Best for: Arbitrary shapes, outlier detection
  ⚠️ Assumes: Density-based clusters
  🎯 Use when: Noise in data, unknown cluster shapes

🔹 GMM (Gaussian Mixture):
  ✅ Best for: Overlapping clusters, soft clustering
  ⚠️ Assumes: Gaussian distributions
  🎯 Use when: Probabilistic assignments needed

🔹 Selection Tips:
  • Start with K-Means for baseline
  • Use DBSCAN for noisy/irregular data
  • Try Hierarchical for exploratory analysis
  • Use GMM for probabilistic clustering
"""
    print(insights)

# ================================
# PRACTICAL DEMONSTRATION
# ================================

def clustering_demo():
    """Complete clustering demonstration"""
    print("🚀 CLUSTERING ALGORITHMS DEMO")
    print("=" * 35)
    
    # 1. Create datasets
    datasets = create_sample_datasets()
    
    # 2. Analyze each dataset
    for dataset_key, (X, dataset_name) in datasets.items():
        
        # Find optimal clusters (for structured data)
        if dataset_key != 'circles':  # Skip for circular data
            optimal_k, k_range, inertias, silhouette_scores = find_optimal_clusters(X)
        else:
            optimal_k = 2  # We know circles have 2 clusters
        
        # Compare all algorithms
        results, X_scaled = compare_clustering_algorithms(X, dataset_name, optimal_k)
        
        # Create visualization
        create_clustering_visualization(results, X_scaled, dataset_name)
    
    # 3. Show insights
    show_clustering_insights()
    
    print("\n✅ CLUSTERING ANALYSIS COMPLETE!")
    print("🎯 What you learned in 15 minutes:")
    print("  • K-Means for spherical clusters")
    print("  • Hierarchical for cluster hierarchy")
    print("  • DBSCAN for irregular shapes & outliers")
    print("  • GMM for probabilistic clustering")
    print("  • Evaluation with silhouette scores")
    
    print("\n🚀 NEXT STEPS (Practice):")
    print("  1. Try different cluster numbers")
    print("  2. Experiment with DBSCAN parameters")
    print("  3. Apply to your own datasets")
    print("  4. Combine with dimensionality reduction")

if __name__ == "__main__":
    clustering_demo()