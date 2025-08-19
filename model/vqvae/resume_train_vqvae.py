import torch
import os
from model.base import BaseVAE
from torch import nn
from torch.nn import functional as F
from types_ import *
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image, make_grid
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from model.vq_vae import VQVAE

grayscale = True
checkpoint_path = "best_vqvae_model_0.0007.pth"
resume_epoch = 30

def train_vqvae(data_loader, model, optimizer, num_epochs=150, log_interval=5):
    model.train()
    print("Starting Training")
    writer = SummaryWriter('/home/aiworker1/GAN/runs/VQVAE')
    best_loss = float('inf')
    start_epoch = 0

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        print("Checkpoint keys:", checkpoint.keys())

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_loss = checkpoint.get("best_loss", float("inf"))
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print("Checkpoint does not contain full training state. Loading model weights only.")
            model.load_state_dict(checkpoint)  # Load directly
            start_epoch = resume_epoch  # Restart training
    
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        for i, (images, _) in enumerate(data_loader):
            images = images.to(device)
            optimizer.zero_grad()
            recon_images, input_images, vq_loss = model(images)
            loss_dict = model.loss_function(recon_images, input_images, vq_loss)
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(data_loader) + i)

        avg_loss = total_loss / len(data_loader)
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        print(f'Epoch {epoch+1}, Loss: {avg_loss}')
        
        # Save best model and checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
                }, f'best_vqvae_model_{best_loss:.4f}_final.pth')
            print(f'Saved best model with loss: {best_loss}')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }, f'best_vqvae_model_{best_loss:.4f}_final.pth')

        if epoch % log_interval == 0:
            model.eval()
            with torch.no_grad():
                sample_images, _ = next(iter(data_loader))
                sample_images = sample_images.to(device)
                recon_images, _, _ = model(sample_images)
                sample_images = (sample_images + 1) / 2
                recon_images = (recon_images + 1) / 2
                input_grid = make_grid(sample_images[:16], nrow=4, normalize=True)
                recon_grid = make_grid(recon_images[:16], nrow=4, normalize=True)
                writer.add_image('Input Images', input_grid, epoch)
                writer.add_image('Reconstructed Images', recon_grid, epoch)
            model.train()
    writer.close()

image_size = 512
if grayscale:
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),  
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),  
        transforms.Normalize([0.5], [0.5]),  
    ])
    out = 1
else:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),  
        transforms.Normalize([0.5], [0.5]),  
    ])
    out = 3

dataset = ImageFolder(root='images', transform=transform)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vqvae = VQVAE(in_channels=1, out=out, embedding_dim=512, num_embeddings=512).to(device)
optimizer = Adam(vqvae.parameters(), lr=1e-3)

train_vqvae(data_loader, vqvae, optimizer, num_epochs=100)

vqvae.eval()
with torch.no_grad():
    for images, _ in data_loader:
        images = images.to(device)
        recon_images, _, _ = vqvae(images)
        recon_images = (recon_images + 1) / 2
        save_image(recon_images, 'reconstructed_images_vqvae.png', nrow=4)
        break
