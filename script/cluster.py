import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sklearn.cluster import KMeans
from model import VQVAE
from dataset import ImageDataset
from torch.utils.data import DataLoader
from config import parse
from torchvision import transforms
import numpy as np
import csv


def get_embeddings(data_loader, model, device):
    print("Extracting embeddings...")
    model.eval()
    embeddings = []
    file_paths = []
    
    with torch.no_grad():
        for data in data_loader:
            inputs, paths = data  # Your dataset returns (image, img_path)
            inputs = inputs.to(device)
            
            # Get model outputs - adjust for your VQVAE/VAE
            encoded_features = model.encode(inputs)[0] # Adjust based on your model output
            
            pooled_features = torch.nn.functional.adaptive_avg_pool2d(encoded_features, (1, 1))
            embeddings_flat = pooled_features.view(pooled_features.size(0), -1)
            
            embeddings.append(embeddings_flat.cpu())
            file_paths.extend(paths)

            # Collect file paths (already provided by your dataset)
            file_paths.extend(paths)
            
    
    return torch.cat(embeddings), file_paths

def cluster_embeddings(embeddings, n_clusters=10):
    print(f"Clustering embeddings into {n_clusters} clusters...")
    print(f"Embeddings shape: {embeddings.shape}")
    
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.numpy()
    else:
        embeddings_np = embeddings
    
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(embeddings_np)
    return clusters

def main():
    args = parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use args parameters instead of hardcoded values
    model = VQVAE(in_channels=1,
                  out = 1,
                  num_embeddings=args.hidden_dim, 
                  embedding_dim=args.latent_dim,
                  img_size=512).to(device)
    
    checkpoint_path = os.path.join(args.exp, "best_vqvae_model.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    
    transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((512, 512)),  # Ensure this matches your training image size
    transforms.ToTensor(),
])

    dataset = ImageDataset(args.data_dir, train_flag=True, transforms=None)

    # Dataloaders
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0 
    )

    embeddings, file_paths = get_embeddings(dataloader, model, device)
    clusters = cluster_embeddings(embeddings)


    np.save(os.path.join(args.exp, "embeddings.npy"), embeddings.numpy())
    np.save(os.path.join(args.exp, "clusters.npy"), clusters)

    # Save file paths and cluster labels to a CSV
    csv_path = os.path.join(args.exp, "cluster_labels.csv")
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file_path', 'cluster_label'])
        for path, label in zip(file_paths, clusters):
            writer.writerow([path, label])

if __name__ == '__main__':
    main()