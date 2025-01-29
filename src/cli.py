import torch
import yaml
import os
from pathlib import Path

from data.loader import get_dataloader
from models.model import CNN
from models.trainer import CNNTrainer
from models.tester import Tester

from torch import nn
from torchmetrics import Accuracy
from torchinfo import summary


from torchviz import make_dot
import graphviz

import mlflow

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main training pipeline."""

    # Config parameters
    num_epochs = int(os.sys.argv[1])
    config = os.sys.argv[2]
    cfg = load_config(config)
    model_name = cfg['model']['name']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    learning_rate=cfg['training']['learning_rate'],
    loss_fn=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

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
    
    # # Create a sample input (assuming 1 second of audio at 22050Hz)
    # sample_input = torch.randn(1, 22050)

    # # Generate the visualization
    # dot = make_dot(model(sample_input), params=dict(model.named_parameters()))

    # # Set some visualization parameters
    # dot.attr(rankdir='TB')  # Top to bottom layout
    # dot.attr('node', shape='box')  # Box shaped nodes

    # # Save the visualization
    # dot.render("output/cnn_architecture", format="png", cleanup=True)

    # print("Architecture visualization has been saved as 'cnn_architecture.png'")
    
    # Initialize trainer
    trainer = CNNTrainer(
        model=model,
        learning_rate=learning_rate,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr=learning_rate,
        device=device
    )

    metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)

    with mlflow.start_run():
        params = {
            "epochs": num_epochs,
            "learning_rate": cfg['training']['learning_rate'],
            "batch_size": cfg['training']['batch_size'],
            "loss_function": loss_fn.__class__.__name__,
            "metric_function": metric_fn.__class__.__name__,
            "optimizer": optimizer.__class__.__name__
        }

        # Log training parameters.
        mlflow.log_params(params)

        # Log model summary.
        with open("model_summary.txt", "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_summary.txt")

        # for t in range(num_epochs):
        #     print(f"Epoch {t+1}\n-------------------------------")
        #     train(train_dataloader, model, loss_fn, metric_fn, optimizer)
        
        # Train model
        trainer.train(
            train_loader=train_loader,
            valid_loader=valid_loader,
            num_epochs=num_epochs,
            save_path=save_path
        )
    
        # print(f"\nTraining completed. Model saved to {save_path}")

        # Save the trained model to MLflow.
        mlflow.pytorch.log_model(model, "model")
    

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