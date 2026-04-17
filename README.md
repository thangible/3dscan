#VQVAE / SimCLR Embedding & Clustering

Small toolkit to train image embedders (VQVAE or SimCLR), produce embeddings, run clustering analyses and create visualization artifacts and an interactive dashboard.

## Features

- Train a SimCLR embedder (ResNet50 backbone + projection head) or a VQVAE.
- Extract embeddings and run K-Means / DBSCAN sweeps with evaluation metrics.
- Prepare PCA / t-SNE visualization datasets and launch an interactive Dash dashboard.
- WandB integration for training metrics and reconstruction image logging.

## Requirements

- Python 3.10+
- PyTorch (matching your CUDA or CPU setup)
- torchvision, numpy, scikit-learn, matplotlib, plotly, dash, wandb, lightly, tqdm
- Install packages from the project's `pyproject.toml` using Astral UV's `uv sync` (recommended).
  1. Install `uv` (if you don't have it):
     Visit https://docs.astral.sh/uv/ and follow the installation instructions for your platform.

  2. Sync dependencies from `pyproject.toml` (PowerShell example):
     pwsh.exe -Command "uv sync"

If you don’t have a requirements file, use the project’s pyproject.toml to reproduce environment.

## Quickstart (recommended pipeline)

1. Prepare a folder with your images and note the path (e.g. C:\data\images).

2. Train SimCLR embeddings (recommended):

python script\train_embedder_simclr.py --data_dir <path_to_images> --exp exp --batch_size 64 --input_dim 224 --max_epoch_num 100 --lr 1e-4

This creates `exp/best_simclr_model.pth` and logs to Weights & Biases.

3. Cluster analysis and evaluation (SimCLR embeddings):

python script\cluster_analysis_simclr.py --data_dir <path_to_images> --exp exp --batch_size 128 --input_dim 224

Outputs are stored under `exp/clustering_results/` (raw/scaled embeddings, csvs, summary JSON, plots).

4. Prepare visualization data (PCA + t-SNE):

python script\prepare_visualization_data.py

This writes `exp/clustering_results/visualization/visualization_data_*.npy` files.

5. Launch interactive dashboard:

pwsh.exe -Command "python script\dashboard.py"

Open http://127.0.0.1:8050 in your browser and select a visualization file.

## Important options / notes

- Use `--exp` to set the output experiment folder; many artifacts are written under `exp/<exp>/`.
- `--input_dim` must match the image size used at training time.
- Scripts use `num_workers=0` by default for Windows compatibility.
- SimCLR expects 3-channel input; grayscale images are repeated to 3 channels in the augmentation pipeline.
- WandB logs: training metrics, checkpoints, and (for VQVAE) side-by-side input vs reconstruction images every 5 epochs.

## Project structure (high level)

- script/ — training, clustering, visualization and dashboard scripts
  - train_embedder_simclr.py — train SimCLR embedder
  - cluster_analysis_simclr.py — extract embeddings, run KMeans/DBSCAN sweeps, evaluate
  - train_embedder.py — VQVAE training + reconstruction logging
  - cluster.py — simple VQVAE embedding extraction + KMeans
  - prepare_visualization_data.py — PCA/t-SNE and produce .npy visualization artifacts
  - dashboard.py — Dash app to explore visualizations and images
- model/ — VQVAE and related model code
- dataset/ — ImageDataset and augmentation utilities
- exp/ — experiment outputs (checkpoints, embeddings, clustering_results, visualization)
- config/ — argument parsing and configuration (args.py)

## Reproduce exact commands / inspect args

Run any script with `--help` to see full argument list and defaults, for example:

pwsh.exe -Command "python script\train_embedder_simclr.py --help"

## Troubleshooting

- OOM: reduce `--batch_size` or `--input_dim`.
- Missing checkpoint: ensure `--exp` points to the directory used for training.
- Dashboard shows no visualizations: run `prepare_visualization_data.py --all` and confirm files under `exp/clustering_results/visualization/`.
