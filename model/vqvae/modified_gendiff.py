import os
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import UNet2DModel, DDPMScheduler, PNDMScheduler

# ==================== CONFIGURATION ====================
IMAGE_SIZE = 256
GRID_SIZE = 2  # 2x2 grid
BATCH_SIZE = 4  # Number of images generated simultaneously
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "diffusion_512.pth"  # Path to trained model
OUTPUT_FOLDER = "generated_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Select Scheduler: True = PNDM (fast), False = DDPM (slow)
USE_PNDM = False  # Toggle this to switch between PNDM and DDPM

# ==================== LOAD MODEL ====================
model = UNet2DModel(
    sample_size=IMAGE_SIZE,
    in_channels=3,
    out_channels=3,
    layers_per_block=3,
    block_out_channels=(64, 128, 256, 512),
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ==================== CHOOSE SCHEDULER ====================
if USE_PNDM:
    scheduler = PNDMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(50)  # PNDM works best with fewer steps (~25)
    print("Using PNDM Scheduler (fast sampling)")
else:
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    print("Using DDPM Scheduler (slow but accurate)")

# Fix potential device issues
scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(DEVICE)

# ==================== GENERATE IMAGES ====================
print("Generating images...")

total_images = GRID_SIZE * GRID_SIZE
num_batches = (total_images + BATCH_SIZE - 1) // BATCH_SIZE  # Calculate number of batches

generated_images = []
for batch_idx in range(num_batches):
    batch_size = min(BATCH_SIZE, total_images - batch_idx * BATCH_SIZE)  # Adjust for last batch
    sample = torch.randn((batch_size, 3, IMAGE_SIZE, IMAGE_SIZE), device=DEVICE)

    with torch.no_grad():
        if USE_PNDM:
            for t in scheduler.timesteps:
                timesteps = torch.full((batch_size,), t, device=DEVICE, dtype=torch.long)
                noise_pred = model(sample, timesteps)["sample"]
                sample = scheduler.step(noise_pred, t, sample).prev_sample
        else:
            for t in reversed(range(scheduler.config.num_train_timesteps)):
                timesteps = torch.full((batch_size,), t, device=DEVICE, dtype=torch.long)
                noise_pred = model(sample, timesteps)["sample"]
                sample = scheduler.step(noise_pred, t, sample).prev_sample

    # Save the generated images
    for i in range(batch_size):
        img = (sample[i].clamp(-1, 1) + 1) / 2  # Convert back to [0,1]
        img = T.ToPILImage()(img)
        img_idx = batch_idx * BATCH_SIZE + i
        img.save(os.path.join(OUTPUT_FOLDER, f"generated_{img_idx}.png"))
        generated_images.append(img)

# ==================== DISPLAY IMAGE GRID ====================
print("Creating image grid...")
fig, axes = plt.subplots(GRID_SIZE, GRID_SIZE, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(generated_images[i])
    ax.axis("off")
plt.tight_layout()
plt.savefig("grid.png")

print(f"Generated images saved in: {OUTPUT_FOLDER}")