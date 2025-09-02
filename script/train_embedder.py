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
from training import set_randomness, get_augmentations
from model import VQVAE
from config.args import parse
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils


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


def log_reconstruction_images(model, dataloader, device, epoch, num_images=8):
    """
    Log input and reconstructed images to wandb for visualization
    """
    model.eval()
    print(f"Logging reconstruction images for epoch {epoch}...")
    
    with torch.no_grad():
        # Get a batch of images
        data_iter = iter(dataloader)
        inputs, _ = next(data_iter)
        inputs = inputs[:num_images].to(device)
        
        # Get reconstructions
        outputs = model(inputs)
        reconstructions = outputs[0] # First output is usually the reconstruction
        
        # Convert tensors to numpy and denormalize
        inputs_np = inputs.cpu()
        reconstructions_np = reconstructions.cpu()
        
        # Denormalize from [-1, 1] to [0, 1]
        inputs_np = (inputs_np + 1) / 2
        reconstructions_np = (reconstructions_np + 1) / 2
        
        # Clamp values to [0, 1]
        inputs_np = torch.clamp(inputs_np, 0, 1)
        reconstructions_np = torch.clamp(reconstructions_np, 0, 1)
        
        # Create side-by-side comparison
        comparison_images = []
        
        for i in range(num_images):
            # Get single images
            input_img = inputs_np[i]
            recon_img = reconstructions_np[i]
            
            # Create side-by-side image
            if input_img.shape[0] == 1:  # Grayscale
                input_img = input_img.squeeze(0)
                recon_img = recon_img.squeeze(0)
                
                # Create side-by-side comparison
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
                ax1.imshow(input_img, cmap='gray')
                ax1.set_title('Input')
                ax1.axis('off')
                
                ax2.imshow(recon_img, cmap='gray')
                ax2.set_title('Reconstructed')
                ax2.axis('off')
                
                plt.tight_layout()
                
                # Convert plot to wandb image
                comparison_images.append(wandb.Image(plt, caption=f"Sample {i+1}"))
                plt.close()
        
        # Log images to wandb
        wandb.log({
            f"reconstructions/epoch_{epoch}": comparison_images,
            "reconstructions/epoch": epoch
        })
    
    model.train()


def train_vae(model, dataloader, optimizer, scheduler, device, checkpoint, args, num_epochs=20):
    """
    Train VAE model with support for both VanillaVAE and VQVAE
    """
    # Initialize static variable for best loss
    if not hasattr(train_vae, 'best_loss'):
        train_vae.best_loss = float('inf')
    
    # Create exp directory if it doesn't exist
    os.makedirs("exp", exist_ok=True)
    
    model.train()
    
    # Main epoch progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        total_loss = 0
        total_recons_loss = 0
        total_kld_loss = 0
        
        # Batch progress bar
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", leave=False)
        
        for batch_idx, data in enumerate(batch_pbar):
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
            
            # Update batch progress bar with current loss
            batch_pbar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'ReconLoss': f'{loss_dict.get("Reconstruction_Loss", 0).item():.4f}',
                'VQ/KLD': f'{loss_dict.get("KLD", loss_dict.get("VQ_Loss", 0)).item():.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log batch-level metrics to wandb (every 100 batches)
            if batch_idx % 100 == 0:
                wandb.log({
                    "batch/loss": total_loss_batch.item(),
                    "batch/reconstruction_loss": loss_dict.get('Reconstruction_Loss', 0).item() if 'Reconstruction_Loss' in loss_dict else 0,
                    "batch/kld_or_vq_loss": loss_dict.get('KLD', loss_dict.get('VQ_Loss', 0)).item(),
                    "batch/learning_rate": optimizer.param_groups[0]['lr'],
                    "batch/epoch": epoch,
                    "batch/batch_idx": batch_idx
                })
        
        # Close batch progress bar
        batch_pbar.close()
        
        # Step scheduler
        scheduler.step()
        
        # Calculate average losses
        avg_loss = total_loss / len(dataloader)
        avg_recons_loss = total_recons_loss / len(dataloader)
        avg_kld_loss = total_kld_loss / len(dataloader)
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'AvgLoss': f'{avg_loss:.4f}',
            'ReconLoss': f'{avg_recons_loss:.4f}',
            'VQ/KLD': f'{avg_kld_loss:.4f}',
            'BestLoss': f'{train_vae.best_loss:.4f}'
        })
        
        # Log reconstruction images every 5 epochs
        if (epoch) % 5 == 0:
            print(f"\nLogging reconstruction images for epoch {epoch + 1}...")
            log_reconstruction_images(model, dataloader, device, epoch + 1)
        
        # Log epoch-level metrics to wandb
        wandb.log({
            "epoch/loss": avg_loss,
            "epoch/reconstruction_loss": avg_recons_loss,
            "epoch/kld_or_vq_loss": avg_kld_loss,
            "epoch/learning_rate": optimizer.param_groups[0]['lr'],
            "epoch/epoch": epoch + 1,
            "epoch/best_loss": train_vae.best_loss
        })
        
        # Log loss to file
        with open("exp/training_log.txt", "a") as log_file:
            log_file.write(f"Epoch {epoch+1},{avg_loss},{avg_recons_loss},{avg_kld_loss}\n")

        # Save best model weights
        if epoch == 0 or avg_loss < train_vae.best_loss:
            train_vae.best_loss = avg_loss
            
            # Save checkpoint with metadata
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': avg_loss,
                'config': {
                    'input_dim': args.input_dim,
                    'hidden_dim': args.hidden_dim,
                    'latent_dim': args.latent_dim,
                    'batch_size': args.batch_size,
                    'lr': args.lr
                }
            }
            torch.save(checkpoint_data, checkpoint)
            
            # # Log model artifact to wandb
            # artifact = wandb.Artifact(f"model-epoch-{epoch+1}", type="model")
            # artifact.add_file(checkpoint)
            # wandb.log_artifact(artifact)
            
            # Update progress bar with best model info
            tqdm.write(f"✓ New best model saved! Loss: {avg_loss:.6f}")
            
            # Log best loss update
            wandb.log({
                "epoch/new_best_loss": avg_loss,
                "epoch/best_epoch": epoch + 1
            })
    
    # Close epoch progress bar
    epoch_pbar.close()
    print("Training completed!")


def main():
    set_randomness()
    args = parse()
    
    # Initialize wandb
    wandb.init(
        project="3d-scanner-vqvae",  # Change to your project name
        name=f"vqvae-{args.exp}",
        config={
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epoch_num,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "experiment": args.exp,
            "data_dir": args.data_dir,
            "model_type": "VQVAE"
        },
        tags=["vqvae", "3d-scanner", "clustering"]
    )
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Log device info
    wandb.log({"system/device": str(device)})
    if torch.cuda.is_available():
        wandb.log({
            "system/gpu_name": torch.cuda.get_device_name(0),
            "system/gpu_memory": torch.cuda.get_device_properties(0).total_memory / 1e9
        })
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((args.input_dim, args.input_dim)),  
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),  
        transforms.Normalize([0.5], [0.5]),  
    ])
    
    print("Loading dataset...")
    train_dataset = ImageDataset(args.data_dir, train_flag=True, transforms=transform)

    # Dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )

    # Log dataset info
    wandb.log({
        "dataset/size": len(train_dataset),
        "dataset/num_batches": len(train_dataloader)
    })
    
    print(f"Dataset loaded: {len(train_dataset)} images, {len(train_dataloader)} batches")

    print("Creating model...")
    model = VQVAE(in_channels=1, 
                  out=1,
                  embedding_dim=args.hidden_dim, 
                  num_embeddings=args.latent_dim,
                  img_size=512).to(device)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log({
        "model/total_parameters": total_params,
        "model/trainable_parameters": trainable_params
    })
    
    print(f"Model created: {total_params:,} total parameters, {trainable_params:,} trainable")

    # Optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(model, args)
    
    # Training loop
    checkpoint_path = os.path.join(args.exp, "best_vqvae_model.pth")
    
    # Check if resuming from checkpoint
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint.get('epoch', 0)
            train_vae.best_loss = checkpoint.get('best_loss', float('inf'))
            
            wandb.log({
                "resume/checkpoint_epoch": start_epoch,
                "resume/checkpoint_loss": checkpoint.get('best_loss', float('inf'))
            })
    
    # Watch model for gradients (optional - can be memory intensive)
    # wandb.watch(model, log="all", log_freq=100)
    
    print("Starting training...")
    train_vae(model, train_dataloader, optimizer, scheduler, device, checkpoint_path, args, num_epochs=args.max_epoch_num)
    
    # Finish wandb run
    wandb.finish()

if __name__ == '__main__':
    main()



