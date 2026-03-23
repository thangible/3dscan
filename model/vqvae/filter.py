import torch
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model.vq_vae import VQVAE  # Ensure this is correctly imported
from tqdm import tqdm  # For progress tracking

# Parameters
out = 1
image_size = 512
benchmark_images_folder = "/home/aiworker1/GAN/images"
synth_image_folder = "/home/aiworker1/GAN/generated_images"  # Ensure correct subfolder
output_dir = "/home/aiworker1/GAN/vqvae_filtered_images"
filter_factor = 40
batch_size = 32

from PIL import Image
import os

dataset_path = "/home/aiworker1/GAN/generated_images/synthetics"

for file in os.listdir(dataset_path):
    try:
        with Image.open(os.path.join(dataset_path, file)) as img:
            img.verify()  # Check if the image is corrupted
    except Exception as e:
        print(f"Corrupt image detected and removed: {file}")
        os.remove(os.path.join(dataset_path, file))

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vqvae = VQVAE(in_channels=1, out=out, embedding_dim=512, num_embeddings=512).to(device)
checkpoint = torch.load("best_vqvae_model_0.0003_final.pth", map_location=device)
vqvae.load_state_dict(checkpoint["model_state_dict"])  # Load model weights
vqvae.eval()

# Define dataset transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((image_size, image_size)),  
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),  
    transforms.Normalize([0.5], [0.5]),  
])

# Load datasets
real_dataset = ImageFolder(benchmark_images_folder, transform=transform)
generated_dataset = ImageFolder(synth_image_folder, transform=transform)

real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
generated_loader = DataLoader(generated_dataset, batch_size=batch_size, shuffle=False)

# Compute mean reconstruction error for real images
real_losses = []
with torch.no_grad():
    for images, _ in tqdm(real_loader, desc="Processing Real Images"):
        images = images.to(device)
        recons, _, vq_loss = vqvae(images)

        # Compute loss per image (not per batch)
        per_image_losses = torch.nn.functional.mse_loss(recons, images, reduction='none')
        per_image_losses = per_image_losses.view(per_image_losses.shape[0], -1).mean(dim=1)

        real_losses.extend(per_image_losses.cpu().tolist())  # Convert to list

# Compute threshold
threshold = torch.tensor(real_losses).mean() * filter_factor
print(f"Quality Threshold: {threshold:.6f}")

# Prepare to filter generated images
filtered_images = []
generated_filenames = sorted(os.listdir(f"{synth_image_folder}/synthetics"))  # Ensure sorted order
generated_losses = []

# Filter generated images based on threshold
with torch.no_grad():
    for idx, (images, _) in enumerate(tqdm(generated_loader, desc="Filtering Generated Images")):
        images = images.to(device)
        recons, _, vq_loss = vqvae(images)

        # Compute loss per image
        per_image_losses = torch.nn.functional.mse_loss(recons, images, reduction='none')
        per_image_losses = per_image_losses.view(per_image_losses.shape[0], -1).mean(dim=1)

        batch_start = idx * batch_size  # Compute actual index offset
        for j, loss_value in enumerate(per_image_losses):
            generated_losses.append(loss_value.item())

            if loss_value < threshold:
                filtered_images.append(generated_filenames[batch_start + j])  # Correct filename mapping

# Compute quality of generated images
quality = torch.tensor(generated_losses).mean()
print(f"Quality of generated images: {quality:.6f}")

# Save filtered images
os.makedirs(output_dir, exist_ok=True)

for img_name in filtered_images:
    img_path = os.path.join(synth_image_folder, "synthetics", img_name)
    save_path = os.path.join(output_dir, img_name)
    
    if os.path.exists(img_path):  # Ensure file exists before moving
        os.rename(img_path, save_path)

print(f"Filtered {len(filtered_images)} high-quality generated images.")
