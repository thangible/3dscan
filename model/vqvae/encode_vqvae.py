import torch
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model.vq_vae import VQVAE  # Ensure this imports your trained VQVAE model

# Parameters
image_folder = "images"  # Change this to your image folder path
latent_save_folder = "latents"  # Change this to where you want to save the latents
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Ensure this matches your training image size
    transforms.ToTensor(),
])

# Load dataset
dataset = ImageFolder(root=image_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Load trained model
model = VQVAE(in_channels=3, embedding_dim=512, num_embeddings=512)  # Adjust based on your model config
model.load_state_dict(torch.load("best_vqvae_model.pth", map_location=device))
model.to(device)
model.eval()

# Ensure save directory exists
os.makedirs(latent_save_folder, exist_ok=True)

# Extract and save latents
with torch.no_grad():
    for i, (images, _) in enumerate(dataloader):
        images = images.to(device)
        latents = model.encode(images)[0]  # Get latent vectors
        save_path = os.path.join(latent_save_folder, f"latents_batch_{i}.pt")
        torch.save(latents.cpu(), save_path)
        print(f"Saved: {save_path}")

print("Latent extraction complete!")
