import argparse
import yaml
import os
from pathlib import Path

from data.loader import get_dataloader
from models.model import CNN
from models.trainer import CNNTrainer

# def parse_args():
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(description='Train CNN model on GTZAN dataset')
#     parser.add_argument('--num_epochs', type=int, required=True,
#                         help='Number of epochs to train')
#     parser.add_argument('--config', type=str, default='config.yaml',
#                         help='Path to configuration file')
    
#     return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main training pipeline."""
    num_epochs = int(os.sys.argv[1])
    config = os.sys.argv[2]
    cfg = load_config(config)
    
    # Create save directory if it doesn't exist
    save_dir = Path(cfg['output']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'best_model.ckpt'
        
    # Initialize dataloaders
    train_loader = get_dataloader(
        data_path=cfg['data']['path'],
        split='train',
        num_samples=cfg['data']['num_samples'],
        num_chunks=cfg['data']['num_chunks'],
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['training']['num_workers']
    )
    
    valid_loader = get_dataloader(
        data_path=cfg['data']['path'],
        split='val',
        num_samples=cfg['data']['num_samples'],
        num_chunks=cfg['data']['num_chunks'],
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['training']['num_workers']
    )
    
    # Initialize model
    model = CNN(
        num_channels=cfg['model']['num_channels'],
        sample_rate=cfg['model']['sample_rate'],
        n_fft=cfg['model']['n_fft'],
        num_mels=cfg['model']['num_mels']
    )
    
    # Initialize trainer
    trainer = CNNTrainer(
        model=model,
        learning_rate=cfg['training']['learning_rate'],
    )
    
    # Train model
    valid_losses = trainer.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=num_epochs,
        save_path=save_path
    )
    
    print(f"\nTraining completed. Model saved to {save_path}")