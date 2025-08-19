import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import random
from model.vq_vae import VQVAE

# Load the trained VQVAE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vqvae = VQVAE(in_channels=3, embedding_dim=64, num_embeddings=512).to(device)
vqvae.load_state_dict(torch.load('best_vqvae_model.pth'))
vqvae.eval()

# Data preparation
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
])

# Load the dataset
dataset = ImageFolder(root='images', transform=transform)  # Change to your dataset path

# Randomly sample a subset of indices
num_samples = 32  # Number of random samples to take
random_indices = random.sample(range(len(dataset)), num_samples)
subset = Subset(dataset, random_indices)
data_loader = DataLoader(subset, batch_size=16, shuffle=False)

# Function to calculate MSE
def calculate_mse(original, reconstructed):
    original = original.view(original.size(0), -1)
    reconstructed = reconstructed.view(reconstructed.size(0), -1)
    mse = F.mse_loss(original, reconstructed, reduction='mean')
    return mse.item()

# Save and visualize original and reconstructed images
with torch.no_grad():
    total_mse = 0
    count = 0
    for images, _ in data_loader:
        images = images.to(device)
        recon_images, _, _ = vqvae(images)  # Use VQVAE's forward method

        # Denormalize images for visualization
        images = (images + 1) / 2
        recon_images = (recon_images + 1) / 2

        # Create a grid for visualization
        fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(16, 4))
        for i in range(8):
            axes[0, i].imshow(images[i].cpu().numpy().transpose(1, 2, 0))
            axes[0, i].axis('off')
            axes[1, i].imshow(recon_images[i].cpu().numpy().transpose(1, 2, 0))
            axes[1, i].axis('off')
        plt.savefig('original_vs_reconstructed.png')
        plt.show()

        # Calculate MSE
        total_mse += calculate_mse(images, recon_images)
        count += images.size(0)

    average_mse = total_mse / count
    print(f'Average MSE: {average_mse}')