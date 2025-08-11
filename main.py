from dataset import ImageDataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torchvision import transforms
from model import VAE

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to consistent size
    transforms.ToTensor()
])
img_dir = '../data/normalized_images/normalized_images/*.jpg'
dataset = ImageDataset(img_dir, transform=transform)
dataloader = DataLoader(dataset, 
                       batch_size=32, 
                       shuffle=False,  # Keep order for clustering
                       num_workers=0)


def vae_loss(reconstructed_x, x, mu, log_var):
    reconstruction_loss = F.mse_loss(reconstructed_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconstruction_loss + kl_divergence

def train_vae(model, dataloader, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data in dataloader:
            inputs, _ = data
            optimizer.zero_grad()
            reconstructed_x, mu, log_var = model(inputs)
            loss = vae_loss(reconstructed_x, inputs, mu, log_var)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")
        
        
model = VAE(input_dim=224*224*3, hidden_dim=512, latent_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_vae(model, dataloader, optimizer, num_epochs=20)