import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import ImageDataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
from training import parse, vae_loss, set_randomness
from model import VQVAE


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


def train_vae(model, dataloader, optimizer, device, checkpoint, num_epochs=20):
    """
    Train VAE model with support for both VanillaVAE and VQVAE
    """
    # Initialize static variable for best loss
    if not hasattr(train_vae, 'best_loss'):
        train_vae.best_loss = float('inf')
    
    # Create exp directory if it doesn't exist
    os.makedirs("exp", exist_ok=True)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        total_recons_loss = 0
        total_kld_loss = 0
        
        for batch_idx, data in enumerate(dataloader):
            inputs, _ = data
            inputs = inputs.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - BaseVAE models return different outputs
            outputs = model(inputs)
            
            # Calculate loss using model's loss_function
            loss_dict = model.loss_function(*outputs)
            
            total_loss_batch = loss_dict['loss']
            total_loss_batch.backward()
            optimizer.step()
            
            # Accumulate losses
            total_loss += total_loss_batch.item()
            
            # Handle different loss types (VAE vs VQVAE)
            if 'Reconstruction_Loss' in loss_dict:
                total_recons_loss += loss_dict['Reconstruction_Loss'].item()
            
            if 'KLD' in loss_dict:
                total_kld_loss += loss_dict['KLD'].item()
            elif 'VQ_Loss' in loss_dict:
                total_kld_loss += loss_dict['VQ_Loss'].item()
        
        # Calculate average losses
        avg_loss = total_loss / len(dataloader)
        avg_recons_loss = total_recons_loss / len(dataloader)
        avg_kld_loss = total_kld_loss / len(dataloader)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Total Loss: {avg_loss:.6f}")
        print(f"  Recons Loss: {avg_recons_loss:.6f}")
        if 'KLD' in model.loss_function(*model(inputs.to(device))):
            print(f"  KLD Loss: {avg_kld_loss:.6f}")
        else:
            print(f"  VQ Loss: {avg_kld_loss:.6f}")
        
        # Log loss to file
        with open("exp/training_log.txt", "a") as log_file:
            log_file.write(f"Epoch {epoch+1},{avg_loss},{avg_recons_loss},{avg_kld_loss}\n")

        # Save best model weights
        if epoch == 0 or avg_loss < train_vae.best_loss:
            train_vae.best_loss = avg_loss
            torch.save(model.state_dict(), checkpoint)
            print(f"  New best model saved! Loss: {avg_loss:.6f}")
        
        print("-" * 50)

def main():
    set_randomness()
    args = parse()

    
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((args.input_dim, args.input_dim)),  
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),  
        transforms.Normalize([0.5], [0.5]),  
    ])
    
    train_dataset = ImageDataset(args.data_dir, train_flag=True, transforms=transform)

    # Dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )


    model = VQVAE(in_channels=1, 
                  out=1,
                  hidden_dim=args.hidden_dim, 
                  latent_dim=args.latent_dim).to(device)

    # Optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(model, args)
    
    # Training loop
    checkpoint = os.path.join(args.exp, "best_vae_weights.pth")
    train_vae(model, train_dataloader, optimizer, device, checkpoint, num_epochs=args.max_epoch_num)

if __name__ == '__main__':
    main()



