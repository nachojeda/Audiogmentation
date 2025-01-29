import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from typing import List, Tuple
import mlflow

from torch.utils.data import DataLoader

class CNNTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        loss_function,
        optimizer,
        learning_rate: float = 0.001,
    ):
        """
        Initialize the CNN trainer.
        
        Args:
            model: The CNN model to train
            learning_rate: Learning rate for optimization
            device: Device to run the model on (cuda/cpu)
        """
        self.device = device
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.valid_losses: List[float] = []
        
    def _print_config(self, num_epochs: int) -> None:
        """Print the training configuration."""
        print('\n' + '='*50)
        print('{:^50}'.format('TRAINING CONFIGURATION'))
        print('='*50)
        print('{:<20} : {}'.format('Device', self.device))
        print('{:<20} : {}'.format('Architecture', self.model.__class__.__name__))
        print('{:<20} : {}'.format('Loss Function', self.loss_function.__class__.__name__))
        print('{:<20} : {}'.format('Optimizer', 'Adam'))
        print('{:<20} : {}'.format('Learning Rate', self.learning_rate))
        print('{:<20} : {}'.format('Number of Epochs', num_epochs))
        print('='*50 + '\n')

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Mean training loss for the epoch
        """
        self.model.train()
        losses = []
        
        for wav, genre_index in train_loader:
            wav = wav.to(self.device)
            genre_index = genre_index.to(self.device)

            # Forward pass
            out = self.model(wav)
            loss = self.loss_function(out, genre_index)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            
        return np.mean(losses)

    def _validate(self, valid_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            valid_loader: DataLoader for validation data
            
        Returns:
            Tuple of (validation loss, accuracy)
        """
        self.model.eval()
        y_true = []
        y_pred = []
        losses = []
        
        with torch.no_grad():
            for wav, genre_index in valid_loader:
                wav = wav.to(self.device)
                genre_index = genre_index.to(self.device)

                # Reshape and aggregate chunk-level predictions
                b, c, t = wav.size()
                logits = self.model(wav.view(-1, t))
                logits = logits.view(b, c, -1).mean(dim=1)
                loss = self.loss_function(logits, genre_index)
                losses.append(loss.item())
                _, pred = torch.max(logits.data, 1)

                # Append labels and predictions
                y_true.extend(genre_index.tolist())
                y_pred.extend(pred.tolist())
                
        return np.mean(losses), accuracy_score(y_true, y_pred)

    def _save_model(self, epoch: int, save_path: str) -> None:
        """
        Save the model if it's the best so far.
        
        Args:
            epoch: Current epoch number
            save_path: Path to save the model checkpoint
        """
        if np.argmin(self.valid_losses) == epoch:
            print(f'Saving the best model at {epoch} epochs!')
            torch.save(self.model.state_dict(), save_path)

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        num_epochs: int,
        save_path: str
    ) -> List[float]:
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            valid_loader: DataLoader for validation data
            num_epochs: Number of epochs to train
            save_path: Path to save the best model checkpoint
            
        Returns:
            List of validation losses
        """
        self._print_config(num_epochs)
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self._train_epoch(train_loader)
            print(f'Epoch: [{epoch+1}/{num_epochs}], Train loss: {train_loss:.4f}')

            # Validate
            valid_loss, accuracy = self._validate(valid_loader)
            mlflow.log_metric("loss", f"{valid_loss:3f}", step=epoch)
            mlflow.log_metric("accuracy", f"{accuracy:3f}", step=epoch)

            print(f'Epoch: [{epoch+1}/{num_epochs}], Valid loss: {valid_loss:.4f}, '
                  f'Valid accuracy: {accuracy:.4f}')

            # Save model
            self.valid_losses.append(valid_loss)
            # self._save_model(epoch, save_path)
            