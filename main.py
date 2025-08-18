from dataset.dataset import ImageDataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torchvision import transforms
from model.model import VAE
import random
import numpy as np
import os
from train_parser import parse
from training.loss import vae_loss
from train_util import set_randomness


def setup_optimizer_and_scheduler(model, args):
    """
    Sets up optimizer and scheduler for VAE model.
    """
    # Learning rate and weight decay with defaults
    lr = args.lr if hasattr(args, 'lr') else 1e-3
    weight_decay = args.weight_decay if hasattr(args, 'weight_decay') else 1e-4

    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=lr, weight_decay=weight_decay
    )
    
    # Cosine Annealing Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=args.max_epoch_num, eta_min=1e-5
    )
    
    return optimizer, scheduler


def train_one_epoch(model, dataloader, optimizer, device, epoch, num_epochs):
    model.train()
    total_loss = 0
    
    for batch_idx, (inputs, _) in enumerate(dataloader):
        inputs = inputs.to(device)
        inputs = inputs.view(inputs.size(0), -1)  # Flatten for VAE
        
        optimizer.zero_grad()
        reconstructed_x, mu, log_var = model(inputs)
        loss = vae_loss(reconstructed_x, inputs, mu, log_var)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}")
    return avg_loss


def main():
    set_randomness()
    args = parse()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset
    data_dir = args.data_dir
    
    # Add transforms to resize images to consistent size
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    train_dataset = ImageDataset(data_dir, train_flag=True, transforms=transform)
    val_dataset = ImageDataset(data_dir, train_flag=False, transforms=transform)
    
    # Dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Model
    input_dim = 256 * 256 * 3  # Match the resized image dimensions
    hidden_dim = 512
    latent_dim = 128
    model = VAE(input_dim, hidden_dim, latent_dim).to(device)
    
    # Optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(model, args)
    
    # Training loop
    for epoch in range(1, args.max_epoch_num + 1):
        train_loss = train_one_epoch(model, train_dataloader, optimizer, device, epoch, args.max_epoch_num)
        scheduler.step()


if __name__ == '__main__':
    main()



