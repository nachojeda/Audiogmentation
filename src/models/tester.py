import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import torch

class Tester():
    def __init__(self, model, path) -> None:
        self.model = model
        self.path = path

    def _load_model(self):
        # Load the best model
        S = torch.load(self.path)
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

    def _print_conf_matrix(self, y_true, y_pred, accuracy, GTZAN_GENRES):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, xticklabels=GTZAN_GENRES, yticklabels=GTZAN_GENRES, cmap='YlGnBu')
        print('Accuracy: %.4f' % accuracy)