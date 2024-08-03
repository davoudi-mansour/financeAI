import torch
import torch.optim as optim
import torch.nn as nn

class Finetuner:
    def __init__(self, model, train_dataloader, val_dataloader, num_epochs, learning_rate, device):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def finetune(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            for samples, labels in self.train_dataloader:
                samples, labels = samples.to(self.device), labels.to(self.device).float().unsqueeze(1)

                outputs = self.model(samples)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(self.train_dataloader)
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {train_loss:.4f}')

            # Validation loop
            self.validate()

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for samples, labels in self.val_dataloader:
                samples, labels = samples.to(self.device), labels.to(self.device).float().unsqueeze(1)

                outputs = self.model(samples)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(self.val_dataloader)
        print(f'Validation Loss: {val_loss:.4f}')
