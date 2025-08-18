import argparse
import configargparse

def parse():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='Path to the config file', default = 'config.yaml')
    
    parser.add_argument('--data_dir', type=str, help='Directory containing the dataset', default='../data/normalized_images/normalized_images')
    
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    
    parser.add_argument('--max_epoch_num', type=int, default=20, help='Maximum number of epochs for training')

    return parser.parse_args()