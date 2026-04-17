# 3D Scanner — VQVAE / SimCLR Embedding & Clustering

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
- config/: argument parsing (args.py)

For details and full argument lists, run any script with `--help` (e.g. `python script\train_embedder_simclr.py --help`).
