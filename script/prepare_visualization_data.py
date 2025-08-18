import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import glob

def prepare_visualization_data():
    # Load your existing data
    clusters = np.load("exp/clusters.npy")
    embeddings = np.load("exp/embeddings.npy")
    
    # Get image paths (adjust pattern based on your data structure)
    data_dir = "../data/normalized_images"
    image_paths = []
    
    # Assuming your images are organized in folders or have a consistent naming
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    # Sort to ensure consistent ordering
    image_paths.sort()
    
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
    pca_2d = PCA(n_components=2, random_state=42)
    pca_3d = PCA(n_components=3, random_state=42)
    
    embeddings_2d = pca_2d.fit_transform(embeddings)
    embeddings_3d = pca_3d.fit_transform(embeddings)
    
    # Optional: Also create t-SNE embeddings (takes longer but often better for visualization)
    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30)
    
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
    if not os.path.exists("/exp"):
        os.makedirs("/exp")
    np.save("exp/visualization_data.npy", viz_data)
    print("Visualization data saved to exp/visualization_data.npy")

    return viz_data

if __name__ == "__main__":
    viz_data = prepare_visualization_data()