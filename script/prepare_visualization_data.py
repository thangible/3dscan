import numpy as np
import os
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import glob
import csv
from pathlib import Path
import argparse

def prepare_visualization_data(clustering_results_dir=None, algorithm='kmeans', params=None):
    """
    Prepare visualization data from cluster analysis results
    
    Args:
        clustering_results_dir: Path to clustering results directory
        algorithm: 'kmeans' or 'dbscan'
        params: dict with algorithm parameters (e.g., {'k': 10} or {'eps': 0.5, 'min_samples': 5})
    """
    
    # Set default paths
    if clustering_results_dir is None:
        clustering_results_dir = "exp/clustering_results"
    
    print(f"Loading data from: {clustering_results_dir}")
    
    # Load the clustering summary to find best results if no specific params given
    summary_path = os.path.join(clustering_results_dir, "clustering_summary.json")
    
    if not params and os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # Use best results from summary
        if algorithm == 'kmeans':
            best_result = summary['best_results']['kmeans']
            params = best_result['params']
        elif algorithm == 'dbscan':
            best_result = summary['best_results']['dbscan']
            params = best_result['params']
        
        print(f"Using best {algorithm} parameters: {params}")
    
    # Construct filenames based on algorithm and parameters
    if algorithm == 'kmeans':
        experiment_name = f"kmeans_k{params['k']}"
    elif algorithm == 'dbscan':
        experiment_name = f"dbscan_eps{params['eps']}_min{params['min_samples']}"
    else:
        raise ValueError("Algorithm must be 'kmeans' or 'dbscan'")
    
    # Load embeddings (use scaled embeddings for consistency)
    embeddings_path = os.path.join(clustering_results_dir, "scaled_embeddings.npy")
    if not os.path.exists(embeddings_path):
        # Fallback to raw embeddings
        embeddings_path = os.path.join(clustering_results_dir, "raw_embeddings.npy")
    
    if not os.path.exists(embeddings_path):
        print(f"No embeddings found at {embeddings_path}")
        return None
    
    embeddings = np.load(embeddings_path)
    print(f"Loaded embeddings from: {embeddings_path}")
    
    # Load clustering results
    cluster_csv_path = os.path.join(clustering_results_dir, f"{experiment_name}_results.csv")
    
    if not os.path.exists(cluster_csv_path):
        print(f"Clustering results not found: {cluster_csv_path}")
        return None
    
    image_paths = []
    clusters = []
    is_noise = []
    
    with open(cluster_csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for row in reader:
            image_paths.append(row[0])
            clusters.append(int(row[1]))
            is_noise.append(row[2].lower() == 'true' if len(row) > 2 else False)
    
    clusters = np.array(clusters)
    is_noise = np.array(is_noise)
    
    # Ensure we have the same number of embeddings, clusters, and images
    min_length = min(len(embeddings), len(clusters), len(image_paths))
    embeddings = embeddings[:min_length]
    clusters = clusters[:min_length]
    image_paths = image_paths[:min_length]
    is_noise = is_noise[:min_length]
    
    print(f"Data shapes:")
    print(f"Embeddings: {embeddings.shape}")
    print(f"Clusters: {clusters.shape}")
    print(f"Image paths: {len(image_paths)}")
    print(f"Unique clusters: {len(np.unique(clusters))}")
    if algorithm == 'dbscan':
        print(f"Noise points: {np.sum(is_noise)}")
    
    # Reduce embeddings to 2D and 3D for visualization
    print("Computing PCA projections...")
    pca_2d = PCA(n_components=2, random_state=42)
    pca_3d = PCA(n_components=3, random_state=42)
    
    embeddings_2d = pca_2d.fit_transform(embeddings)
    embeddings_3d = pca_3d.fit_transform(embeddings)
    
    print("Computing t-SNE projections...")
    # For t-SNE, filter out noise points if using DBSCAN for better visualization
    if algorithm == 'dbscan' and np.any(is_noise):
        # Use only non-noise points for t-SNE
        non_noise_mask = ~is_noise
        tsne_embeddings = embeddings[non_noise_mask]
        tsne_perplexity = min(30, max(5, len(tsne_embeddings) // 4))  # Adjust perplexity for smaller datasets
    else:
        tsne_embeddings = embeddings
        tsne_perplexity = min(30, max(5, len(embeddings) // 4))
    
    if len(tsne_embeddings) > 1:
        tsne_2d = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity)
        tsne_3d = TSNE(n_components=3, random_state=42, perplexity=tsne_perplexity)
        
        embeddings_tsne_2d_subset = tsne_2d.fit_transform(tsne_embeddings)
        embeddings_tsne_3d_subset = tsne_3d.fit_transform(tsne_embeddings)
        
        # For DBSCAN with noise, pad the t-SNE results back to full size
        if algorithm == 'dbscan' and np.any(is_noise):
            embeddings_tsne_2d = np.full((len(embeddings), 2), np.nan)
            embeddings_tsne_3d = np.full((len(embeddings), 3), np.nan)
            embeddings_tsne_2d[non_noise_mask] = embeddings_tsne_2d_subset
            embeddings_tsne_3d[non_noise_mask] = embeddings_tsne_3d_subset
        else:
            embeddings_tsne_2d = embeddings_tsne_2d_subset
            embeddings_tsne_3d = embeddings_tsne_3d_subset
    else:
        print("Warning: Not enough non-noise points for t-SNE")
        embeddings_tsne_2d = np.zeros((len(embeddings), 2))
        embeddings_tsne_3d = np.zeros((len(embeddings), 3))
    
    # Create the visualization dataset
    viz_data = {
        'algorithm': algorithm,
        'params': params,
        'experiment_name': experiment_name,
        'embeddings_original': embeddings,
        'embeddings_pca_2d': embeddings_2d,
        'embeddings_pca_3d': embeddings_3d,
        'embeddings_tsne_2d': embeddings_tsne_2d,
        'embeddings_tsne_3d': embeddings_tsne_3d,
        'clusters': clusters,
        'is_noise': is_noise,
        'image_paths': np.array(image_paths),
        'pca_2d_explained_variance': pca_2d.explained_variance_ratio_,
        'pca_3d_explained_variance': pca_3d.explained_variance_ratio_,
        'n_clusters': len(np.unique(clusters)),
        'n_noise_points': np.sum(is_noise) if algorithm == 'dbscan' else 0
    }
    
    # Save the visualization data
    viz_output_dir = os.path.join(clustering_results_dir, "visualization")
    if not os.path.exists(viz_output_dir):
        os.makedirs(viz_output_dir)
    
    viz_filename = f"visualization_data_{experiment_name}.npy"
    viz_path = os.path.join(viz_output_dir, viz_filename)
    np.save(viz_path, viz_data)
    
    print(f"✓ Visualization data saved to {viz_path}")
    print(f"  PCA 2D explained variance: {pca_2d.explained_variance_ratio_.sum():.3f}")
    print(f"  PCA 3D explained variance: {pca_3d.explained_variance_ratio_.sum():.3f}")
    
    return viz_data

def prepare_all_visualizations(clustering_results_dir=None):
    """
    Prepare visualization data for ALL clustering results in one command
    """
    if clustering_results_dir is None:
        clustering_results_dir = "exp/clustering_results"
    
    summary_path = os.path.join(clustering_results_dir, "clustering_summary.json")
    
    if not os.path.exists(summary_path):
        print(f"❌ No clustering summary found at {summary_path}")
        print("Please run cluster_analysis.py first!")
        return
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    print("🚀 Starting visualization preparation for ALL clustering results...")
    print("=" * 60)
    
    success_count = 0
    total_count = 0
    
    # Prepare visualization for all K-means results
    print("\n📊 Preparing K-means visualizations...")
    for result in summary['kmeans_results']:
        if result['valid']:
            total_count += 1
            try:
                prepare_visualization_data(clustering_results_dir, 'kmeans', result['params'])
                success_count += 1
                print(f"✓ K-means k={result['params']['k']} - SUCCESS")
            except Exception as e:
                print(f"❌ K-means k={result['params']['k']} - FAILED: {e}")
    
    # Prepare visualization for all DBSCAN results
    print("\n🔍 Preparing DBSCAN visualizations...")
    for result in summary['dbscan_results']:
        if result['valid'] and result['n_clusters'] > 1:
            total_count += 1
            try:
                prepare_visualization_data(clustering_results_dir, 'dbscan', result['params'])
                success_count += 1
                print(f"✓ DBSCAN eps={result['params']['eps']}, min_samples={result['params']['min_samples']} - SUCCESS")
            except Exception as e:
                print(f"❌ DBSCAN eps={result['params']['eps']}, min_samples={result['params']['min_samples']} - FAILED: {e}")
    
    print("\n" + "=" * 60)
    print(f"🎉 SUMMARY: {success_count}/{total_count} visualizations prepared successfully!")
    
    if success_count > 0:
        viz_dir = os.path.join(clustering_results_dir, "visualization")
        print(f"📁 All visualization files saved in: {viz_dir}")
        print("🚀 You can now run the dashboard with: python dashboard.py")
    else:
        print("❌ No visualizations were prepared. Check your clustering results.")

def prepare_legacy_format():
    """
    Prepare visualization data from legacy format (single cluster_labels.csv and embeddings.npy)
    For backwards compatibility with old dashboard format
    """
    print("📊 Preparing visualization from legacy format...")
    
    # Check if legacy files exist
    if not os.path.exists("exp/cluster_labels.csv"):
        print("❌ Legacy file exp/cluster_labels.csv not found")
        return
    
    if not os.path.exists("exp/embeddings.npy"):
        print("❌ Legacy file exp/embeddings.npy not found")
        return
    
    # Load legacy data
    image_paths = []
    clusters = []
    with open("exp/cluster_labels.csv", newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for row in reader:
            image_paths.append(row[0])
            clusters.append(int(row[1]))
            
    clusters = np.array(clusters)
    embeddings = np.load("exp/embeddings.npy")
     
    # Ensure we have the same number of embeddings, clusters, and images
    min_length = min(len(embeddings), len(clusters), len(image_paths))
    embeddings = embeddings[:min_length]
    clusters = clusters[:min_length]
    image_paths = image_paths[:min_length]
    
    print(f"Data shapes:")
    print(f"Embeddings: {embeddings.shape}")
    print(f"Clusters: {clusters.shape}")
    print(f"Image paths: {len(image_paths)}")
    
    # Reduce embeddings to 2D and 3D for visualization
    print("Computing PCA projections...")
    pca_2d = PCA(n_components=2, random_state=42)
    pca_3d = PCA(n_components=3, random_state=42)
    
    embeddings_2d = pca_2d.fit_transform(embeddings)
    embeddings_3d = pca_3d.fit_transform(embeddings)
    
    # Optional: Also create t-SNE embeddings (takes longer but often better for visualization)
    print("Computing t-SNE projections...")
    perplexity = min(30, len(embeddings) // 4)
    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=perplexity)
    
    embeddings_tsne_2d = tsne_2d.fit_transform(embeddings)
    embeddings_tsne_3d = tsne_3d.fit_transform(embeddings)
    
    # Create the visualization dataset
    viz_data = {
        'embeddings_original': embeddings,
        'embeddings_pca_2d': embeddings_2d,
        'embeddings_pca_3d': embeddings_3d,
        'embeddings_tsne_2d': embeddings_tsne_2d,
        'embeddings_tsne_3d': embeddings_tsne_3d,
        'clusters': clusters,
        'image_paths': np.array(image_paths),
        'pca_2d_explained_variance': pca_2d.explained_variance_ratio_,
        'pca_3d_explained_variance': pca_3d.explained_variance_ratio_
    }
    
    # Save the visualization data
    if not os.path.exists("exp"):
        os.makedirs("exp")
    np.save("exp/visualization_data.npy", viz_data)
    print("✓ Legacy visualization data saved to exp/visualization_data.npy")

    return viz_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare visualization data from clustering results')
    parser.add_argument('--results_dir', type=str, default='exp/clustering_results',
                       help='Path to clustering results directory')
    parser.add_argument('--algorithm', type=str, choices=['kmeans', 'dbscan'], 
                       help='Clustering algorithm to visualize')
    parser.add_argument('--k', type=int, help='K value for K-means')
    parser.add_argument('--eps', type=float, help='EPS value for DBSCAN')
    parser.add_argument('--min_samples', type=int, help='Min samples for DBSCAN')
    parser.add_argument('--all', action='store_true', help='Prepare visualizations for ALL results (recommended)')
    parser.add_argument('--legacy', action='store_true', help='Prepare from legacy format (cluster_labels.csv)')
    
    args = parser.parse_args()
    
    if args.legacy:
        # Use legacy format
        prepare_legacy_format()
    elif args.all:
        # Prepare ALL visualizations (recommended)
        prepare_all_visualizations(args.results_dir)
    elif args.algorithm:
        # Prepare specific algorithm
        if args.algorithm == 'kmeans' and args.k:
            params = {'k': args.k}
        elif args.algorithm == 'dbscan' and args.eps and args.min_samples:
            params = {'eps': args.eps, 'min_samples': args.min_samples}
        else:
            params = None  # Will use best results
        
        viz_data = prepare_visualization_data(args.results_dir, args.algorithm, params)
    else:
        # Default: prepare ALL visualizations
        print("No specific options provided. Preparing ALL visualizations...")
        prepare_all_visualizations(args.results_dir)