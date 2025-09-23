"""
Day 16: Dimensionality Reduction Toolkit 
🎯 Master PCA & t-SNE for visualization 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ================================
# ESSENTIAL DIMENSIONALITY REDUCTION
# ================================

def create_sample_data():
    """Create high-dimensional sample data"""
    print("🔄 Creating sample high-dimensional data...")
    
    # Load digits dataset (64 dimensions)
    digits = load_digits()
    X_digits = digits.data
    y_digits = digits.target
    
    # Create synthetic data (50 dimensions)
    X_synthetic, y_synthetic = make_classification(
        n_samples=800, n_features=50, n_informative=10, 
        n_redundant=5, random_state=42
    )
    
    print(f"✅ Digits data: {X_digits.shape}")
    print(f"✅ Synthetic data: {X_synthetic.shape}")
    
    return (X_digits, y_digits), (X_synthetic, y_synthetic)

def perform_pca_analysis(X, y, dataset_name):
    """Essential PCA analysis with variance explanation"""
    print(f"\n🔍 PCA Analysis: {dataset_name}")
    print("-" * 30)
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA with different components
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    
    pca_10d = PCA(n_components=10)
    X_pca_10d = pca_10d.fit_transform(X_scaled)
    
    # Variance explained
    var_2d = pca_2d.explained_variance_ratio_.sum()
    var_10d = pca_10d.explained_variance_ratio_.sum()
    
    print(f"  📊 2D PCA variance: {var_2d:.3f}")
    print(f"  📊 10D PCA variance: {var_10d:.3f}")
    
    # Evaluate quality with downstream task
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # Original performance
    rf.fit(X_scaled, y)
    original_acc = rf.score(X_scaled, y)
    
    # PCA performance
    rf.fit(X_pca_10d, y)
    pca_acc = rf.score(X_pca_10d, y)
    
    print(f"  🎯 Original accuracy: {original_acc:.3f}")
    print(f"  🎯 PCA accuracy: {pca_acc:.3f}")
    print(f"  📈 Retention ratio: {pca_acc/original_acc:.3f}")
    
    return X_pca_2d, X_pca_10d, pca_2d

def perform_tsne_analysis(X, y, dataset_name):
    """Essential t-SNE analysis for visualization"""
    print(f"\n🎭 t-SNE Analysis: {dataset_name}")
    print("-" * 30)
    
    # Scale and sample data (t-SNE is expensive)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Sample for efficiency
    if len(X_scaled) > 1000:
        indices = np.random.choice(len(X_scaled), 1000, replace=False)
        X_sample = X_scaled[indices]
        y_sample = y[indices]
        print(f"  📝 Sampled {len(indices)} points for efficiency")
    else:
        X_sample = X_scaled
        y_sample = y
    
    # Apply t-SNE
    perplexity = min(30, (len(X_sample) - 1) // 3)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X_sample)
    
    print(f"  📊 Perplexity used: {perplexity}")
    print(f"  📊 KL divergence: {tsne.kl_divergence_:.3f}")
    
    return X_tsne, y_sample

def create_comparison_visualization(results_dict):
    """Create side-by-side comparison of PCA vs t-SNE"""
    print("\n📊 Creating visualization comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('PCA vs t-SNE Comparison', fontsize=14, fontweight='bold')
    
    # Digits dataset
    if 'digits' in results_dict:
        X_pca, X_tsne, y_digits, y_tsne = results_dict['digits']
        
        # PCA - Digits
        scatter1 = axes[0,0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_digits, cmap='tab10', alpha=0.7)
        axes[0,0].set_title('Digits Dataset - PCA')
        axes[0,0].set_xlabel('First Principal Component')
        axes[0,0].set_ylabel('Second Principal Component')
        
        # t-SNE - Digits
        scatter2 = axes[0,1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_tsne, cmap='tab10', alpha=0.7)
        axes[0,1].set_title('Digits Dataset - t-SNE')
        axes[0,1].set_xlabel('t-SNE 1')
        axes[0,1].set_ylabel('t-SNE 2')
    
    # Synthetic dataset
    if 'synthetic' in results_dict:
        X_pca, X_tsne, y_synth, y_tsne = results_dict['synthetic']
        
        # PCA - Synthetic
        scatter3 = axes[1,0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_synth, cmap='viridis', alpha=0.7)
        axes[1,0].set_title('Synthetic Dataset - PCA')
        axes[1,0].set_xlabel('First Principal Component')
        axes[1,0].set_ylabel('Second Principal Component')
        
        # t-SNE - Synthetic
        scatter4 = axes[1,1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_tsne, cmap='viridis', alpha=0.7)
        axes[1,1].set_title('Synthetic Dataset - t-SNE')
        axes[1,1].set_xlabel('t-SNE 1')
        axes[1,1].set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.savefig('dimensionality_reduction_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'dimensionality_reduction_comparison.png'")
    plt.show()

def show_best_practices():
    """Show essential best practices for dimensionality reduction"""
    practices = """
💡 DIMENSIONALITY REDUCTION BEST PRACTICES
==========================================

🔹 PCA (Principal Component Analysis):
  ✅ Use for: Preprocessing, noise reduction, feature compression
  ✅ Linear transformation, preserves global structure
  ✅ Fast and deterministic
  ✅ Components are interpretable (linear combinations)

🔹 t-SNE (t-Distributed Stochastic Neighbor Embedding):
  ✅ Use for: Data visualization, cluster discovery
  ✅ Non-linear, excellent for revealing local structure
  ✅ Great for plotting, not for preprocessing
  ✅ Can be slow and non-deterministic

🔹 When to Use What:
  • Data compression: PCA
  • Visualization: t-SNE (or PCA for speed)
  • Preprocessing: PCA
  • Cluster exploration: t-SNE
  • Large datasets: PCA first, then t-SNE

🔹 Practical Tips:
  • Always scale data before dimensionality reduction
  • For t-SNE: try different perplexity values (5-50)
  • Validate quality with downstream tasks
  • Consider PCA → t-SNE pipeline for large data
"""
    print(practices)

# ================================
# PRACTICAL DEMONSTRATION
# ================================

def dimensionality_reduction_demo():
    """Complete dimensionality reduction demonstration"""
    print("🚀 DIMENSIONALITY REDUCTION DEMO")
    print("=" * 35)
    
    # 1. Create sample data
    (X_digits, y_digits), (X_synthetic, y_synthetic) = create_sample_data()
    
    results = {}
    
    # 2. Analyze digits dataset
    X_pca_digits, _, _ = perform_pca_analysis(X_digits, y_digits, "Handwritten Digits")
    X_tsne_digits, y_tsne_digits = perform_tsne_analysis(X_digits, y_digits, "Handwritten Digits")
    results['digits'] = (X_pca_digits, X_tsne_digits, y_digits, y_tsne_digits)
    
    # 3. Analyze synthetic dataset
    X_pca_synth, _, _ = perform_pca_analysis(X_synthetic, y_synthetic, "Synthetic Data")
    X_tsne_synth, y_tsne_synth = perform_tsne_analysis(X_synthetic, y_synthetic, "Synthetic Data")
    results['synthetic'] = (X_pca_synth, X_tsne_synth, y_synthetic, y_tsne_synth)
    
    # 4. Create visualization
    create_comparison_visualization(results)
    
    # 5. Show best practices
    show_best_practices()
    
    print("\n✅ DIMENSIONALITY REDUCTION COMPLETE!")
    print("🎯 What you learned in 15 minutes:")
    print("  • PCA for compression and preprocessing")
    print("  • t-SNE for visualization and cluster discovery")
    print("  • Quality evaluation with downstream tasks")
    print("  • Best practices for real-world applications")
    
    print("\n🚀 NEXT STEPS (Practice):")
    print("  1. Try different numbers of PCA components")
    print("  2. Experiment with t-SNE perplexity values")
    print("  3. Apply to your own datasets")
    print("  4. Combine PCA → t-SNE for large datasets")

if __name__ == "__main__":
    dimensionality_reduction_demo()