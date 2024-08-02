import torch
import numpy as np

class Tester:
    def __init__(self, model, test_dataloader, device, threshold):
        self.model = model
        self.test_dataloader = test_dataloader
        self.device = device
        self.threshold = threshold

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        fp_reconstruction_errors = []

        with torch.no_grad():
            for samples, labels in self.test_dataloader:
                samples, labels = samples.to(self.device), labels.to(self.device)
                reconstructed, _ = self.model(samples)
                reconstruction_error = torch.mean((reconstructed - samples) ** 2, dim=[1, 2])

                predicted = (reconstruction_error > self.threshold).long()

                # Count correct predictions
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # Compute TP, TN, FP, FN
                TP += ((predicted == 1) & (labels == 1)).sum().item()
                TN += ((predicted == 0) & (labels == 0)).sum().item()
                FP += ((predicted == 1) & (labels == 0)).sum().item()
                FN += ((predicted == 0) & (labels == 1)).sum().item()

                # Collect reconstruction errors for FP samples
                fp_errors = reconstruction_error[(predicted == 1) & (labels == 0)].cpu().numpy()
                fp_reconstruction_errors.extend(fp_errors)

        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')
        print("Reconstruction errors for FP samples:")
        counter = 0
        for error in fp_reconstruction_errors:
            if error > 1.5 * self.threshold:
                counter+=1
        print(counter)
