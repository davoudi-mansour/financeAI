import torch
from dataset import get_dataloaders
from model import LSTMClassifier
from train import Trainer
from test import Tester
from finetune import Finetuner

# Configuration
data_path = 'BinaryClassification/data/BTCUSD_Hourly_Ask_2022.01.01_2024.07.01.csv'
window_length = 16
batch_size = 128
num_epochs = 1000
learning_rate = 0.001
patience = 10
save_path = 'saved_data/best_model.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get dataloaders
train_dataloader, val_dataloader, test_dataloader = get_dataloaders(data_path, window_length, batch_size)

# Model configuration
input_size = 10
hidden_size = 19
num_layers = 2

model = LSTMClassifier(input_size, hidden_size, num_layers).to(device)

# Train the model
trainer = Trainer(model, train_dataloader, val_dataloader, device, num_epochs, learning_rate, patience)
trainer.train()

# Load the best model before testing
model.load_state_dict(torch.load('saved_models/best_classifier.pth'))

# Test the model
tester = Tester(model, test_dataloader, device)
tester.test()

# Finetune the model if needed
finetuner = Finetuner(model, train_dataloader, val_dataloader, num_epochs, learning_rate, device)
finetuner.finetune()
