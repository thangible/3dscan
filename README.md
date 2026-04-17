# VQVAE / SimCLR Embedding & Clustering -

Toolkit to train image embedders (VQVAE or SimCLR), extract embeddings, run clustering, and visualize results with an interactive dashboard.

## Quickstart (concise)

1. Install dependencies (recommended: Astral UV):

   uv sync

2. Train SimCLR (creates `exp/best_simclr_model.pth`):

   python script\train_embedder_simclr.py --data_dir <PATH> --exp <EXP>

3. Run clustering analysis (writes `exp/<EXP>/clustering_results/`):

   python script\cluster_analysis_simclr.py --data_dir <PATH> --exp <EXP>

4. Prepare visualization data (PCA + t-SNE):

   python script\prepare_visualization_data.py --results_dir exp/<EXP>/clustering_results --all

5. Launch dashboard:

   python script\dashboard.py
   Open http://127.0.0.1:8050

## Config file and command-line args

This project supports a YAML config file (default: `config/config.yaml`) and command-line args. Command-line arguments override values in the config file. Pass a custom config with `--config path/to/config.yaml`.

Example `config/config.yaml` (project default):

```yaml
# config/config.yaml
data_dir: ../data/raw_images
input_dim: 512
hidden_dim: 512
latent_dim: 512
batch_size: 512
max_epoch_num: 100
lr: 1e-3
weight_decay: 1e-4
exp: exp
```

Key command-line arguments (see `config/args.py` for full list and defaults):

- `--config` : Path to config file (default: `config/config.yaml`).
- `--data_dir` : Directory containing the dataset (overrides `data_dir` from config).
- `--batch_size` : Batch size for training.
- `--max_epoch_num` : Maximum number of epochs.
- `--input_dim` : Input image size (H=W).
- `--hidden_dim` : Hidden dimension for the model (VAE encoder/decoder channels).
- `--latent_dim` : Latent / embedding dimension.
- `--lr` : Learning rate.
- `--weight_decay` : Weight decay for optimizer.
- `--exp` : Experiment directory to save outputs (overrides config `exp`).

Example: run training using the default config but override batch size and experiment name:

```
python script\train_embedder_simclr.py --config config/config.yaml --batch_size 64 --exp my_experiment
```

## Notes

- Use `--exp` to target the experiment folder (artifacts written under `exp/<EXP>/`).
- Set `--input_dim`, `--batch_size`, etc., as needed (see `--help`).
- Windows: scripts default to `num_workers=0` for compatibility.
- SimCLR expects 3-channel input; grayscale images are repeated to 3 channels internally.
- VQVAE training logs side-by-side input vs reconstruction images to WandB every 5 epochs.

## OLD / full CLI examples (kept for reference)

```

python script\train_embedder_simclr.py --data_dir <path_to_images> --exp exp --batch_size 64 --input_dim 224 --max_epoch_num 100 --lr 1e-4

python script\cluster_analysis_simclr.py --data_dir <path_to_images> --exp exp --batch_size 128 --input_dim 224

python script\prepare_visualization_data.py --results_dir exp/clustering_results --all

python script\dashboard.py
```

## Project layout (short)

- script/: training, clustering, visualization, dashboard
- model/: VQVAE definitions
- dataset/: ImageDataset and augmentations
- exp/: outputs (checkpoints, embeddings, clustering_results, visualization)
- config/: argument parsing and configuration (args.py)

For details and full argument lists, run any script with `--help` (e.g. `python script\train_embedder_simclr.py --help`).

Werkstudent im Max Team, ich unterstürz Max bei entwicklung der
