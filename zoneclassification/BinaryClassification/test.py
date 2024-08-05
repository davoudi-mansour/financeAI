import torch

class Tester:
    def __init__(self, model, test_dataloader, device):
        self.model = model
        self.test_dataloader = test_dataloader
        self.device = device

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        with torch.no_grad():
            for samples, labels in self.test_dataloader:
                samples, labels = samples.to(self.device), labels.to(self.device)
                outputs = self.model(samples)
                predicted = (outputs > 0.5).float().squeeze()  # Use 0.5 as threshold and remove unnecessary dimensions
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Compute TP, TN, FP, FN
                TP += ((predicted == 1) & (labels == 1)).sum().item()
                TN += ((predicted == 0) & (labels == 0)).sum().item()
                FP += ((predicted == 1) & (labels == 0)).sum().item()
                FN += ((predicted == 0) & (labels == 1)).sum().item()

        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        print(f'Total: {total}')
        print(f'True Positives (TP): {TP}')
        print(f'True Negatives (TN): {TN}')
        print(f'False Positives (FP): {FP}')
        print(f'False Negatives (FN): {FN}')
