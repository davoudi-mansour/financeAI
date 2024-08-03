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
        with torch.no_grad():
            for samples, labels in self.test_dataloader:
                samples, labels = samples.to(self.device), labels.to(self.device)
                outputs = self.model(samples)
                predicted = (outputs > 0.5).float()  # Use 0.5 as threshold
                total += labels.size(0)
                correct += (predicted.squeeze() == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        print(total)
