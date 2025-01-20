import torch
import yaml
import os
from pathlib import Path

from data.loader import get_dataloader
from models.model import CNN
from models.trainer import CNNTrainer
from models.tester import Tester

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main training pipeline."""
    num_epochs = int(os.sys.argv[1])
    config = os.sys.argv[2]
    cfg = load_config(config)
    model_name = cfg['model']['name']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create save directory if it doesn't exist
    save_dir = Path(cfg['output']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / (model_name + '.ckpt')
        
    # Initialize dataloaders
    train_loader = get_dataloader(
        data_path=cfg['data']['path'],
        split='train',
        num_samples=cfg['data']['num_samples'],
        num_chunks=cfg['data']['num_chunks'],
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['training']['num_workers'],
        genres=cfg['data']['GTZAN_GENRES']
    )
    
    valid_loader = get_dataloader(
        data_path=cfg['data']['path'],
        split='val',
        num_samples=cfg['data']['num_samples'],
        num_chunks=cfg['data']['num_chunks'],
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['training']['num_workers'],
        genres=cfg['data']['GTZAN_GENRES']
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
        device=device
    )
    
    # Train model
    valid_losses = trainer.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=num_epochs,
        save_path=save_path
    )
    
    print(f"\nTraining completed. Model saved to {save_path}")

    # Evaluate model
    test_loader = get_dataloader(
        data_path=cfg['data']['path'],
        split='test',
        num_samples=cfg['data']['num_samples'],
        num_chunks=cfg['data']['num_chunks'],
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['training']['num_workers'],
        genres=cfg['data']['GTZAN_GENRES']
    )

    tester = Tester(
        model=model,
        path=save_path
    )

    y_true, y_pred, accuracy = tester._evaluate(
        device=device,
        test_loader=test_loader
    )
    tester._print_conf_matrix(
        y_true=y_true,
        y_pred=y_pred,
        accuracy=accuracy,
        GTZAN_GENRES=cfg['data']['GTZAN_GENRES'],
        save_path=('output/'+model_name+'.png')
    )