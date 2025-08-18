import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sklearn.cluster import KMeans
from model import VanillaVAE
from dataset import ImageDataset
from torch.utils.data import DataLoader
from config import parse
from torchvision import transforms
import numpy as np


def get_embeddings(data_loader, model, device):
    print("Extracting embeddings...")
    model.eval()
    embeddings = []
    with torch.no_grad():
        for data in data_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            _, mu, _ = model(inputs)
            embeddings.append(mu.cpu().view(mu.size(0), -1))
    return torch.cat(embeddings)

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
    model = VanillaVAE(args.input_dim, args.hidden_dim, args.latent_dim).to(device)
    checkpoint = os.path.join(args.exp, "best_vae_weights.pth")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    
    transform = transforms.Compose([
        transforms.Resize((args.input_dim, args.input_dim)),
        transforms.ToTensor(),
    ])

    dataset = ImageDataset(args.data_dir, train_flag=True, transforms=transform)

    # Dataloaders
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 
    )

    embeddings = get_embeddings(dataloader, model, device)
    clusters = cluster_embeddings(embeddings)


    np.save(os.path.join(args.exp, "embeddings.npy"), embeddings.numpy())
    np.save(os.path.join(args.exp, "clusters.npy"), clusters)

if __name__ == '__main__':
    main()