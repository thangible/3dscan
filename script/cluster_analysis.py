import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from model import VQVAE
from dataset import ImageDataset
from torch.utils.data import DataLoader
from config import parse
from torchvision import transforms
import numpy as np
import csv
import json
import matplotlib.pyplot as plt
from datetime import datetime
import pickle


class ClusterAnalyzer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.results_dir = os.path.join(args.exp, "clustering_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Standard K-means parameters
        self.kmeans_k_values = [2, 3, 5, 8, 10, 15, 20, 25, 30, 50]
        
        # Standard DBSCAN parameters
        self.dbscan_eps_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        self.dbscan_min_samples = [3, 5, 10, 15, 20]
        
    def get_embeddings(self, data_loader, model):
        """Extract embeddings from the model"""
        print("Extracting embeddings...")
        model.eval()
        embeddings = []
        file_paths = []
        
        with torch.no_grad():
            for batch_idx, data in enumerate(data_loader):
                if batch_idx % 10 == 0:
                    print(f"Processing batch {batch_idx}/{len(data_loader)}")
                    
                inputs, paths = data
                inputs = inputs.to(self.device)
                
                # Get model outputs - adjust for your VQVAE
                encoded_features = model.encode(inputs)[0]
                
                # Global average pooling
                pooled_features = torch.nn.functional.adaptive_avg_pool2d(encoded_features, (1, 1))
                embeddings_flat = pooled_features.view(pooled_features.size(0), -1)
                
                embeddings.append(embeddings_flat.cpu())
                file_paths.extend(paths)
        
        embeddings = torch.cat(embeddings)
        print(f"Extracted embeddings shape: {embeddings.shape}")
        return embeddings, file_paths
    
    def preprocess_embeddings(self, embeddings):
        """Standardize embeddings for clustering"""
        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.numpy()
        else:
            embeddings_np = embeddings
            
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_np)
        
        # Save scaler for future use
        scaler_path = os.path.join(self.results_dir, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
            
        return embeddings_scaled, scaler
    
    def evaluate_clustering(self, embeddings, labels, algorithm_name, params):
        """Evaluate clustering quality using multiple metrics"""
        if len(set(labels)) < 2:  # Need at least 2 clusters
            return {
                'algorithm': algorithm_name,
                'params': params,
                'n_clusters': len(set(labels)),
                'n_noise': sum(labels == -1) if -1 in labels else 0,
                'silhouette_score': None,
                'calinski_harabasz_score': None,
                'davies_bouldin_score': None,
                'valid': False
            }
        
        # Remove noise points for silhouette score (DBSCAN)
        if -1 in labels:
            mask = labels != -1
            clean_embeddings = embeddings[mask]
            clean_labels = labels[mask]
        else:
            clean_embeddings = embeddings
            clean_labels = labels
        
        if len(set(clean_labels)) < 2:
            silhouette = None
        else:
            silhouette = silhouette_score(clean_embeddings, clean_labels)
        
        try:
            calinski = calinski_harabasz_score(clean_embeddings, clean_labels)
            davies_bouldin = davies_bouldin_score(clean_embeddings, clean_labels)
        except:
            calinski = None
            davies_bouldin = None
        
        return {
            'algorithm': algorithm_name,
            'params': params,
            'n_clusters': len(set(labels)),
            'n_noise': sum(labels == -1) if -1 in labels else 0,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski,
            'davies_bouldin_score': davies_bouldin,
            'valid': True
        }
    
    def run_kmeans_analysis(self, embeddings, file_paths):
        """Run K-means with different k values"""
        print("Running K-means analysis...")
        kmeans_results = []
        
        for k in self.kmeans_k_values:
            print(f"Running K-means with k={k}")
            
            # Fit K-means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            # Evaluate
            evaluation = self.evaluate_clustering(embeddings, labels, 'kmeans', {'k': k})
            kmeans_results.append(evaluation)
            
            # Save results
            self.save_clustering_results(labels, file_paths, f"kmeans_k{k}")
            
            # Save model
            model_path = os.path.join(self.results_dir, f"kmeans_k{k}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(kmeans, f)
        
        return kmeans_results
    
    def run_dbscan_analysis(self, embeddings, file_paths):
        """Run DBSCAN with different parameters"""
        print("Running DBSCAN analysis...")
        dbscan_results = []
        
        for eps in self.dbscan_eps_values:
            for min_samples in self.dbscan_min_samples:
                print(f"Running DBSCAN with eps={eps}, min_samples={min_samples}")
                
                # Fit DBSCAN
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(embeddings)
                
                # Evaluate
                evaluation = self.evaluate_clustering(
                    embeddings, labels, 'dbscan', 
                    {'eps': eps, 'min_samples': min_samples}
                )
                dbscan_results.append(evaluation)
                
                # Save results if valid clustering
                if evaluation['valid'] and evaluation['n_clusters'] > 1:
                    self.save_clustering_results(
                        labels, file_paths, 
                        f"dbscan_eps{eps}_min{min_samples}"
                    )
        
        return dbscan_results
    
    def save_clustering_results(self, labels, file_paths, experiment_name):
        """Save clustering results to CSV and numpy files"""
        # Save labels as numpy array
        labels_path = os.path.join(self.results_dir, f"{experiment_name}_labels.npy")
        np.save(labels_path, labels)
        
        # Save detailed CSV
        csv_path = os.path.join(self.results_dir, f"{experiment_name}_results.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['file_path', 'cluster_label', 'is_noise'])
            for path, label in zip(file_paths, labels):
                is_noise = label == -1
                writer.writerow([path, label, is_noise])
    
    def plot_evaluation_metrics(self, kmeans_results, dbscan_results):
        """Create plots for clustering evaluation"""
        # K-means elbow plot
        plt.figure(figsize=(15, 5))
        
        # Subplot 1: K-means metrics
        plt.subplot(1, 3, 1)
        k_values = [r['params']['k'] for r in kmeans_results if r['valid']]
        silhouette_scores = [r['silhouette_score'] for r in kmeans_results if r['valid']]
        
        plt.plot(k_values, silhouette_scores, 'bo-', label='Silhouette Score')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('K-means: Silhouette Score vs k')
        plt.grid(True)
        plt.legend()
        
        # Subplot 2: DBSCAN cluster counts
        plt.subplot(1, 3, 2)
        valid_dbscan = [r for r in dbscan_results if r['valid'] and r['n_clusters'] > 1]
        if valid_dbscan:
            eps_values = [r['params']['eps'] for r in valid_dbscan]
            n_clusters = [r['n_clusters'] for r in valid_dbscan]
            
            plt.scatter(eps_values, n_clusters, alpha=0.6)
            plt.xlabel('EPS Value')
            plt.ylabel('Number of Clusters')
            plt.title('DBSCAN: Clusters vs EPS')
            plt.grid(True)
        
        # Subplot 3: Comparison of best results
        plt.subplot(1, 3, 3)
        best_kmeans = max([r for r in kmeans_results if r['valid']], 
                         key=lambda x: x['silhouette_score'] or 0)
        best_dbscan = max([r for r in dbscan_results if r['valid']], 
                         key=lambda x: x['silhouette_score'] or 0)
        
        algorithms = ['K-means', 'DBSCAN']
        scores = [best_kmeans['silhouette_score'], best_dbscan['silhouette_score']]
        
        plt.bar(algorithms, scores)
        plt.ylabel('Best Silhouette Score')
        plt.title('Best Algorithm Comparison')
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'clustering_evaluation.png'), dpi=300)
        plt.close()
    
    def save_summary_report(self, kmeans_results, dbscan_results, embeddings_info):
        """Save comprehensive summary report"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'embeddings_info': embeddings_info,
            'kmeans_results': kmeans_results,
            'dbscan_results': dbscan_results,
            'best_results': {
                'kmeans': max([r for r in kmeans_results if r['valid']], 
                            key=lambda x: x['silhouette_score'] or 0),
                'dbscan': max([r for r in dbscan_results if r['valid']], 
                            key=lambda x: x['silhouette_score'] or 0)
            }
        }
        
        # Save as JSON
        summary_path = os.path.join(self.results_dir, 'clustering_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create readable text report
        report_path = os.path.join(self.results_dir, 'clustering_report.txt')
        with open(report_path, 'w') as f:
            f.write("CLUSTERING ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of samples: {embeddings_info['n_samples']}\n")
            f.write(f"Embedding dimension: {embeddings_info['embedding_dim']}\n\n")
            
            f.write("BEST RESULTS:\n")
            f.write("-" * 20 + "\n")
            best_kmeans = summary['best_results']['kmeans']
            best_dbscan = summary['best_results']['dbscan']
            
            f.write(f"Best K-means: k={best_kmeans['params']['k']}, "
                   f"Silhouette={best_kmeans['silhouette_score']:.3f}\n")
            f.write(f"Best DBSCAN: eps={best_dbscan['params']['eps']}, "
                   f"min_samples={best_dbscan['params']['min_samples']}, "
                   f"Silhouette={best_dbscan['silhouette_score']:.3f}\n\n")
            
            f.write("All results saved in clustering_summary.json\n")


def main():
    args = parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize analyzer
    analyzer = ClusterAnalyzer(args, device)
    
    # Load model
    model = VQVAE(in_channels=1,
                  out=1,
                  num_embeddings=args.hidden_dim, 
                  embedding_dim=args.latent_dim,
                  img_size=512).to(device)
    
    checkpoint_path = os.path.join(args.exp, "best_vqvae_model.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Setup data
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    dataset = ImageDataset(args.data_dir, train_flag=True, transforms=transform)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 
    )
    
    # Extract embeddings
    embeddings, file_paths = analyzer.get_embeddings(dataloader, model)
    
    # Save raw embeddings
    embeddings_path = os.path.join(analyzer.results_dir, "raw_embeddings.npy")
    np.save(embeddings_path, embeddings.numpy())
    
    # Preprocess embeddings
    embeddings_scaled, scaler = analyzer.preprocess_embeddings(embeddings)
    
    # Save scaled embeddings
    scaled_embeddings_path = os.path.join(analyzer.results_dir, "scaled_embeddings.npy")
    np.save(scaled_embeddings_path, embeddings_scaled)
    
    # Save file paths
    paths_file = os.path.join(analyzer.results_dir, "file_paths.txt")
    with open(paths_file, 'w') as f:
        for path in file_paths:
            f.write(f"{path}\n")
    
    # Run clustering analyses
    kmeans_results = analyzer.run_kmeans_analysis(embeddings_scaled, file_paths)
    dbscan_results = analyzer.run_dbscan_analysis(embeddings_scaled, file_paths)
    
    # Create evaluation plots
    analyzer.plot_evaluation_metrics(kmeans_results, dbscan_results)
    
    # Save comprehensive report
    embeddings_info = {
        'n_samples': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'preprocessing': 'StandardScaler applied'
    }
    analyzer.save_summary_report(kmeans_results, dbscan_results, embeddings_info)
    
    print(f"\nAnalysis complete! Results saved in: {analyzer.results_dir}")
    print("Files generated:")
    print("- raw_embeddings.npy")
    print("- scaled_embeddings.npy") 
    print("- clustering_summary.json")
    print("- clustering_report.txt")
    print("- clustering_evaluation.png")
    print("- Individual clustering results for each parameter combination")


if __name__ == '__main__':
    main()