import argparse
import configargparse

def parse():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='Path to the config file', default = 'config.yaml')
    
    parser.add_argument('--data_dir', type=str, help='Directory containing the dataset', default='../data/normalized_images/normalized_images')
    
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    
    parser.add_argument('--max_epoch_num', type=int, default=20, help='Maximum number of epochs for training')
    
    parser.add_argument('--input_dim', type=int, default=64, help='Input dimension for the images')
    
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for the VAE')
    
    parser.add_argument('--latent_dim', type=int, default=32, help='Latent dimension for the VAE')
    
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for the optimizer')

    return parser.parse_args()