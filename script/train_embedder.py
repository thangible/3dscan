from dataset.dataset import ImageDataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
from training.train_parser import parse
from training.loss import vae_loss
from training.train_util import set_randomness

from model.vanilla_vae import VanillaVAE


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


def train_vae(model, dataloader, optimizer, device, num_epochs=20):
    # Initialize static variable for best loss
    if not hasattr(train_vae, 'best_loss'):
        train_vae.best_loss = float('inf')
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data in dataloader:
            inputs, _ = data
            inputs = inputs.to(device)  # Move inputs to the correct device
            optimizer.zero_grad()
            reconstructed_x, mu, log_var = model(inputs)
            loss = vae_loss(reconstructed_x, inputs, mu, log_var)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")
        
        avg_loss = total_loss / len(dataloader)
        # Log loss to file
        with open("training_log.txt", "a") as log_file:
            log_file.write(f"Epoch {epoch+1},{avg_loss}\n")

        # Save best model weights
        if epoch == 0 or avg_loss < train_vae.best_loss:
            train_vae.best_loss = avg_loss
            torch.save(model.state_dict(), "best_vae_weights.pth")

def main():
    set_randomness()
    args = parse()

    
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
      
    # Add transforms to resize images to consistent size
    transform = transforms.Compose([
        transforms.Resize((args.input_dim, args.input_dim)),
        transforms.ToTensor(),
    ])
    
    train_dataset = ImageDataset(args.data_dir, train_flag=True, transforms=transform)
    val_dataset = ImageDataset(args.data_dir, train_flag=False, transforms=transform)
    
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
    

    model = VanillaVAE(args.input_dim, args.hidden_dim, args.latent_dim).to(device)

    # Optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(model, args)
    
    # Training loop
    train_vae(model, train_dataloader, optimizer, device, num_epochs=args.max_epoch_num)

        
        


if __name__ == '__main__':
    main()



