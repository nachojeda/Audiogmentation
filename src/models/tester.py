import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import matplotlib.pyplot as plt

class Tester():
    def __init__(self, model, path) -> None:
        self.model = model
        self.path = path

    def _load_model(self):
        # Load the best model
        S = torch.load(self.path, weights_only=True)
        self.model.load_state_dict(S)

    def _evaluate(self, device, test_loader):
        
        # Load model
        self._load_model()

        # Run evaluation
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for wav, genre_index in test_loader:
                wav = wav.to(device)
                genre_index = genre_index.to(device)

                # reshape and aggregate chunk-level predictions
                b, c, t = wav.size()
                logits = self.model(wav.view(-1, t))
                logits = logits.view(b, c, -1).mean(dim=1)
                _, pred = torch.max(logits.data, 1)

                # append labels and predictions
                y_true.extend(genre_index.tolist())
                y_pred.extend(pred.tolist())


        accuracy = accuracy_score(y_true, y_pred)

        return y_true, y_pred, accuracy

    def _print_conf_matrix(self, y_true, y_pred, accuracy, GTZAN_GENRES, save_path=None):
        """
        Create and optionally save a confusion matrix visualization
        
        Parameters:
            y_true: array-like of true labels
            y_pred: array-like of predicted labels
            accuracy: float, classification accuracy
            GTZAN_GENRES: list of genre labels
            save_path: str, optional path to save the plot (e.g., 'confusion_matrix.png')
        """
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        
        # Create heatmap
        sns.heatmap(cm, 
                    annot=True,
                    fmt='d',  # Show numbers as integers
                    xticklabels=GTZAN_GENRES, 
                    yticklabels=GTZAN_GENRES,
                    cmap='YlGnBu')
        
        # Add labels and title
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'Confusion matrix saved to: {save_path}')
        
        
        print('Accuracy: %.4f' % accuracy)

        # Display the plot
        plt.show()

