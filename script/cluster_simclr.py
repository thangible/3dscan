import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sklearn.cluster import KMeans
#from model import VQVAE
from dataset import ImageDataset
from torch.utils.data import DataLoader
from config import parse
from torchvision import transforms
import numpy as np
import csv
import torchvision.models as models
from lightly.models.modules import SimCLRProjectionHead
from torchvision.models import ResNet50_Weights


def get_embeddings(data_loader, backbone, projection_head, device, use_projection=False):
    print("Extracting embeddings from backbone...")
    backbone.eval()
    if projection_head is not None:
        projection_head.eval()

    embeddings = []
    file_paths = []

    with torch.no_grad():
        for data in data_loader:
            inputs, paths = data  # dataset returns (image, img_path)
            inputs = inputs.to(device)

            # forward through backbone
            h = backbone(inputs)

            # If backbone leaves spatial dims, pool to (1,1)
            if h.dim() == 4:
                h = torch.nn.functional.adaptive_avg_pool2d(h, (1, 1))
            embeddings_flat = torch.flatten(h, start_dim=1)

            if use_projection and projection_head is not None:
                z = projection_head(embeddings_flat)
                embeddings.append(z.cpu())
            else:
                embeddings.append(embeddings_flat.cpu())

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

    # Build ResNet backbone and projection head (shapes must match training)
    resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    backbone = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)

    projection_head = SimCLRProjectionHead(2048, 2048, 128).to(device)

    # Look for SimCLR checkpoint (saved by train_embedder_simclr)
    checkpoint_path = os.path.join(args.exp, "best_simclr_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}.\nRun training or provide the correct checkpoint path.")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load backbone/projection state dicts if present (support older 'model_state_dict')
    if isinstance(checkpoint, dict):
        if 'backbone_state_dict' in checkpoint:
            backbone.load_state_dict(checkpoint['backbone_state_dict'])
        elif 'model_state_dict' in checkpoint:
            # try to load into backbone if shapes permit (best-effort)
            try:
                backbone.load_state_dict(checkpoint['model_state_dict'], strict=False)
            except Exception:
                print("Warning: could not load 'model_state_dict' into backbone with strict=False")

        if 'projection_state_dict' in checkpoint:
            projection_head.load_state_dict(checkpoint['projection_state_dict'])
        elif 'model_state_dict' in checkpoint:
            # projection head may not be present in older checkpoints
            pass
    else:
        raise ValueError("Unexpected checkpoint format")

    # Dataset transforms: ensure images are 3-channel for ResNet
    transform = transforms.Compose([
        transforms.Resize((args.input_dim, args.input_dim)),
        transforms.ToTensor(),
        # If dataset is grayscale, expand to 3 channels
        transforms.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
        # Normalize; use 0.5 center if that was used during training
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = ImageDataset(args.data_dir, train_flag=True, transforms=transform)

    # Dataloaders
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Extract embeddings from backbone (optionally projection head)
    use_projection = getattr(args, 'use_projection_for_clustering', False)
    embeddings, file_paths = get_embeddings(dataloader, backbone, projection_head, device, use_projection=use_projection)

    clusters = cluster_embeddings(embeddings, n_clusters=getattr(args, 'n_clusters', 10))

    os.makedirs(args.exp, exist_ok=True)
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