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
from lightly.models.modules import SimCLRProjectionHead
from lightly.loss import NTXentLoss
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from config.args import parse


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
    # Augmentation pipeline tuned for microstructure images.
    # Move ToPILImage here and repeat grayscale to 3 channels inside the pipeline.
    aug_pipeline = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop((args.input_dim, args.input_dim), scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    

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
                # apply augmentation pipeline (ToPILImage inside pipeline handles tensors)
                cpu_img = img.cpu()
                v1 = aug_pipeline(cpu_img)
                v2 = aug_pipeline(cpu_img)
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
        project="3d-scanner-simclr",
        name=f"simclr-{args.exp}",
        config={
            "input_dim": args.input_dim,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epoch_num,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "experiment": args.exp,
            "data_dir": args.data_dir,
            "model_type": "SimCLR"
        },
        tags=["simclr", "3d-scanner", "clustering"]
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
    
    # Minimal dataset transform: resize and convert to tensor. Augmentations and normalization
    # (including channel repeat) are applied inside the SimCLR augmentation pipeline.
    transform = transforms.Compose([
        transforms.Resize((args.input_dim, args.input_dim)),
        transforms.ToTensor(),
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
    resnet = models.resnet50(weight = ResNet50_Weights.IMAGENET1K_V1)
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
    checkpoint_path = os.path.join(args.exp, "best_simclr_model.pth")
    
    # Check if resuming from checkpoint (load backbone + projection head + opt/scheduler)
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        cp = torch.load(checkpoint_path, map_location=device)
        if isinstance(cp, dict):
            if 'backbone_state_dict' in cp:
                backbone.load_state_dict(cp['backbone_state_dict'])
            if 'projection_state_dict' in cp:
                projection_head.load_state_dict(cp['projection_state_dict'])
            if 'optimizer_state_dict' in cp:
                optimizer.load_state_dict(cp['optimizer_state_dict'])
            if 'scheduler_state_dict' in cp:
                scheduler.load_state_dict(cp['scheduler_state_dict'])
            start_epoch = cp.get('epoch', 0)
            train_simclr.best_loss = cp.get('best_loss', float('inf'))

            wandb.log({
                "resume/checkpoint_epoch": start_epoch,
                "resume/checkpoint_loss": cp.get('best_loss', float('inf'))
            })
    
    # Watch model for gradients (optional - can be memory intensive)
    # wandb.watch(model, log="all", log_freq=100)
    
    print("Starting SimCLR training...")
    train_simclr(backbone, projection_head, train_dataloader, optimizer, scheduler, device, checkpoint_path, args, num_epochs=args.max_epoch_num)
     
    # Finish wandb run
    wandb.finish()

if __name__ == '__main__':
    main()



