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
import torchvision.models as models
from lightly.models.modules import SimCLRProjectionHead
from lightly.loss import NTXentLoss


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


def train_simclr(backbone, projection_head, dataloader, optimizer, scheduler, device, checkpoint, args, num_epochs=20):
    """
    Simple SimCLR training loop. Expects a dataloader that yields single images; two augmented views
    are generated on-the-fly per batch.
    """
    # Initialize static variable for best loss
    if not hasattr(train_simclr, 'best_loss'):
        train_simclr.best_loss = float('inf')

    os.makedirs("exp", exist_ok=True)

    backbone.train()
    projection_head.train()

    criterion = NTXentLoss(temperature=getattr(args, 'temperature', 0.1))

    # augmentation pipeline used to create two views from each image
    aug_pipeline = transforms.Compose([
        transforms.RandomResizedCrop((args.input_dim, args.input_dim), scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # helpers to convert tensor->PIL->augmented tensor
    to_pil = transforms.ToPILImage()

    epoch_pbar = tqdm(range(num_epochs), desc="SimCLR Training", unit="epoch")

    for epoch in epoch_pbar:
        total_loss = 0.0
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", leave=False)

        for batch_idx, data in enumerate(batch_pbar):
            inputs, _ = data  # inputs: (B, C, H, W) tensor

            # build two augmented views per sample
            batch_view1 = []
            batch_view2 = []
            for img in inputs:
                pil = to_pil(img.cpu())
                v1 = aug_pipeline(pil)
                v2 = aug_pipeline(pil)
                batch_view1.append(v1)
                batch_view2.append(v2)

            x1 = torch.stack(batch_view1).to(device)
            x2 = torch.stack(batch_view2).to(device)

            optimizer.zero_grad()

            # forward through backbone (remove trailing spatial dims if present)
            h1 = backbone(x1)
            h2 = backbone(x2)

            # flatten if necessary
            h1 = torch.flatten(h1, start_dim=1)
            h2 = torch.flatten(h2, start_dim=1)

            z1 = projection_head(h1)
            z2 = projection_head(h2)

            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            batch_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

            # batch-level wandb logging (every 100 batches)
            if batch_idx % 100 == 0:
                wandb.log({
                    "batch/simclr_loss": loss.item(),
                    "batch/learning_rate": optimizer.param_groups[0]['lr'],
                    "batch/epoch": epoch,
                    "batch/batch_idx": batch_idx
                })

        batch_pbar.close()
        scheduler.step()

        avg_loss = total_loss / max(1, len(dataloader))
        epoch_pbar.set_postfix({
            'AvgLoss': f'{avg_loss:.4f}',
            'BestLoss': f'{train_simclr.best_loss:.4f}'
        })

        wandb.log({
            "epoch/simclr_loss": avg_loss,
            "epoch/learning_rate": optimizer.param_groups[0]['lr'],
            "epoch/epoch": epoch + 1,
            "epoch/best_loss": train_simclr.best_loss
        })

        with open("exp/training_log.txt", "a") as log_file:
            log_file.write(f"Epoch {epoch+1},{avg_loss}\n")

        # Save best model (projection head + optional backbone)
        if epoch == 0 or avg_loss < train_simclr.best_loss:
            train_simclr.best_loss = avg_loss

            checkpoint_data = {
                'epoch': epoch + 1,
                'projection_state_dict': projection_head.state_dict(),
                'backbone_state_dict': backbone.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': avg_loss,
                'config': {
                    'input_dim': args.input_dim,
                    'batch_size': args.batch_size,
                    'lr': args.lr
                }
            }
            torch.save(checkpoint_data, checkpoint)
            tqdm.write(f"✓ New best SimCLR model saved! Loss: {avg_loss:.6f}")
            wandb.log({
                "epoch/new_best_simclr_loss": avg_loss,
                "epoch/best_epoch": epoch + 1
            })

    epoch_pbar.close()
    print("SimCLR training completed!")


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
    # Build a ResNet backbone and a SimCLR projection head
    resnet = models.resnet50(pretrained=True)
    backbone = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)  # remove final fc

    # Optionally freeze backbone unless fine-tuning
    finetune_backbone = getattr(args, 'finetune', False)
    for p in backbone.parameters():
        p.requires_grad = finetune_backbone

    projection_head = SimCLRProjectionHead(2048, 2048, 128).to(device)

    # Simple wrapper for compatibility with downstream code
    model = None

    # Log model info (backbone + projection head)
    total_params = sum(p.numel() for p in backbone.parameters()) + sum(p.numel() for p in projection_head.parameters())
    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad) + sum(p.numel() for p in projection_head.parameters() if p.requires_grad)
    wandb.log({
        "model/total_parameters": total_params,
        "model/trainable_parameters": trainable_params
    })
    
    print(f"Model created: {total_params:,} total parameters, {trainable_params:,} trainable")

    # Optimizer and scheduler
    # Optimize projection head and optionally backbone
    params_to_optimize = list(projection_head.parameters())
    if finetune_backbone:
        params_to_optimize += list(backbone.parameters())

    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.max_epoch_num, eta_min=1e-5)
 
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
    
    print("Starting SimCLR training...")
    train_simclr(backbone, projection_head, train_dataloader, optimizer, scheduler, device, checkpoint_path, args, num_epochs=args.max_epoch_num)
    
    # Finish wandb run
    wandb.finish()

if __name__ == '__main__':
    main()



