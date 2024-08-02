import torch
import torch.optim as optim
import torch.nn as nn

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, device, num_epochs=100, learning_rate=0.001, patience=30):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            for samples, labels in self.train_dataloader:
                samples, labels = samples.to(self.device), labels.to(self.device)

                # Train only on samples with label 0
                mask = labels == 0
                samples = samples[mask]
                if len(samples) == 0:
                    continue
                self.optimizer.zero_grad()
                reconstructed, latent = self.model(samples)
                loss = self.criterion(reconstructed, samples)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(self.train_dataloader)
            val_loss = self.validate()
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Save the best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'saved_models/best_autoencoder.pth')
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve == self.patience:
                    print("Early stopping!")
                    break

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for samples, labels in self.val_dataloader:
                samples, labels = samples.to(self.device), labels.to(self.device)

                # Validate only on samples with label 0
                mask = labels == 0
                samples = samples[mask]
                if len(samples) == 0:
                    continue

                reconstructed, _ = self.model(samples)
                loss = self.criterion(reconstructed, samples)
                val_loss += loss.item()

        val_loss /= len(self.val_dataloader)
        return val_loss
