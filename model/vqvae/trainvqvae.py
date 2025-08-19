import torch
from model.base import BaseVAE
from torch import nn
from torch.nn import functional as F
from types_ import *
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from model.vq_vae import VQVAE
from torchvision.utils import make_grid  # Add this import

grayscale = True

def train_vqvae(data_loader, model, optimizer, num_epochs=150, log_interval=5):
    model.train()
    print("Starting Training")
    writer = SummaryWriter('/home/aiworker1/GAN/runs/VQVAE')
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (images, _) in enumerate(data_loader):
            images = images.to(device)
            optimizer.zero_grad()
            recon_images, input_images, vq_loss = model(images)
            # recon_images = recon_images.mean(dim=1, keepdim=True)  # Convert to grayscale
            loss_dict = model.loss_function(recon_images, input_images, vq_loss)
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Log training loss per batch
            writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(data_loader) + i)

        # Log average loss per epoch
        avg_loss = total_loss / len(data_loader)
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        print(f'Epoch {epoch+1}, Loss: {avg_loss}')
        
        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'best_vqvae_model_{best_loss:.4f}.pth')
            print(f'Saved best model with loss: {best_loss}')
        
        if epoch % log_interval == 0:
            print(f"trained Epoch: {epoch} with loss of: {avg_loss}")

        if epoch % log_interval == 0:
            model.eval()
            with torch.no_grad():
                sample_images, _ = next(iter(data_loader))  # Get a batch of images
                sample_images = sample_images.to(device)
                recon_images, _, _ = model(sample_images)

                # Denormalize images for visualization
                sample_images = (sample_images + 1) / 2
                recon_images = (recon_images + 1) / 2

                # Create image grids
                input_grid = make_grid(sample_images[:16], nrow=4, normalize=True)
                recon_grid = make_grid(recon_images[:16], nrow=4, normalize=True)

                # Log images to TensorBoard
                writer.add_image('Input Images', input_grid, epoch)
                writer.add_image('Reconstructed Images', recon_grid, epoch)

            model.train()  # Switch back to training mode

    writer.close()

image_size = 512

if grayscale:
    # Data preparation
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((image_size, image_size)),  
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),  
        transforms.Normalize([0.5], [0.5]),  
    ])
    out = 1
else:
    # Data preparation
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),  
        transforms.Normalize([0.5], [0.5]),  
    ])
    out = 3

dataset = ImageFolder(root='images', transform=transform)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Model initialization and training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vqvae = VQVAE(in_channels=1, out=out, embedding_dim=512, num_embeddings=512).to(device)
optimizer = Adam(vqvae.parameters(), lr=1e-3)

train_vqvae(data_loader, vqvae, optimizer, num_epochs=100)

# Save a few sample outputs
vqvae.eval()
with torch.no_grad():
    for images, _ in data_loader:
        images = images.to(device)
        recon_images, _, _ = vqvae(images)
        # Denormalize before saving
        recon_images = (recon_images + 1) / 2
        save_image(recon_images, 'reconstructed_images_vqvae.png', nrow=4)
        break