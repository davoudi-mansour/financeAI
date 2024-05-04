import torch
from torch.utils.data import Dataset
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def min_max_normalize(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data
class FinancialTimeSeriesDataset(Dataset):

    def __init__(self, data, window_length, zone_area, stop_area, indexes=None):
        self.data = data
        self.window_length = window_length
        self.zone_area = zone_area
        self.stop_area = stop_area
        self.indexes = indexes

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        start_idx = self.indexes[idx]
        sequence = self.data[start_idx:start_idx + self.window_length]
        last_price = sequence[-1]
        candidate = start_idx + self.window_length

        while candidate < len(self.data):
            price_change = self.data[candidate] - last_price
            if abs(price_change) >= self.zone_area:
                val_sequence = self.data[start_idx + self.window_length:candidate]
                state = 0 if price_change >= self.zone_area else 1
                if self.validate(self.stop_area, last_price, val_sequence, state):
                    label = 0 if price_change >= self.zone_area else 2
                else:
                    label = 1
                normalized_sequence = min_max_normalize(sequence)
                return torch.tensor(normalized_sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
            candidate += 1
        return torch.zeros(self.window_length), torch.tensor(1)  # In case no valid data found

    def validate(self, stop_area, last_price, val_sequence, state):
        flag = True
        if state == 0:
            for item in val_sequence:
                if item - last_price < - stop_area:
                    flag = False
        else:
            for item in val_sequence:
                if item - last_price > stop_area:
                    flag = False
        return flag

NUM_CLASSES = 3
class CNNBiLSTM(nn.Module):
    def __init__(self, dropout_rate=0.3, num_classes=NUM_CLASSES):
        super(CNNBiLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.batch_norm = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.lstm = nn.LSTM(input_size=32, hidden_size=16, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(32, num_classes)  # Output size changed to num_classes for multi-class classification

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        output, _ = self.lstm(x)
        x = output[:, -1, :]
        x = self.fc(x)
        x = torch.softmax(x, dim=1)  # Apply softmax activation for multi-class classification
        return x

# Initialize model, loss function, and optimizer
model = CNNBiLSTM().cuda()  # Move model to GPU
criterion = nn.CrossEntropyLoss()  # Cross Entropy Loss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.01)

_15min_samples = np.load('data/_5min_samples.npy')
zone_area = 0.0038
stop_area = zone_area / 2
window_length = 10000

# Calculate total number of sequences
total_sequences = len(range(0, len(_15min_samples) - window_length, 10))
train_size = int(total_sequences * 0.9)
test_size = total_sequences - train_size
sequences = range(0, len(_15min_samples) - window_length, 10)
# Generate indices for training and testing
indices = list(sequences)
np.random.shuffle(indices)
train_indices = indices[:train_size]
test_indices = indices[train_size:]

# Create training and testing datasets
train_dataset = FinancialTimeSeriesDataset(_15min_samples, window_length, zone_area, stop_area, indexes=train_indices)
test_dataset = FinancialTimeSeriesDataset(_15min_samples, window_length, zone_area, stop_area, indexes=test_indices)

# Create DataLoaders for each dataset
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Early stopping parameters
patience = 15
best_val_loss = float('inf')
best_model_path = None
epochs_without_improvement = 0

# Lists to store training and validation losses
train_losses = []
val_losses = []

# Directory to save models
save_dir = "saved_models"

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for sequences, labels in train_dataloader:
        sequences, labels = sequences.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(sequences.unsqueeze(1))
        loss = criterion(outputs, labels)  # Loss calculation changed for multi-class classification
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_train_loss = running_loss / len(train_dataloader)
    train_losses.append(epoch_train_loss)
    print(f"Epoch {epoch+1}, Training Loss: {epoch_train_loss}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_sequences, val_labels in val_dataloader:
            val_sequences, val_labels = val_sequences.cuda(), val_labels.cuda()
            val_outputs = model(val_sequences.unsqueeze(1))
            val_loss += criterion(val_outputs, val_labels).item()  # Loss calculation changed for multi-class classification
    val_loss /= len(val_dataloader)
    val_losses.append(val_loss)
    print(f"Validation Loss: {val_loss}")

    # Check for early stopping and save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        if best_model_path:
            os.remove(best_model_path)
        best_model_path = os.path.join(save_dir, f"best_model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), best_model_path)
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print("Early stopping.")
            break

# Plotting the loss values
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('results/loss.png')
plt.show()

# Load the best model state
if best_model_path:
    model.load_state_dict(torch.load(best_model_path))

# Evaluation loop
model.eval()
class_correct = [0] * NUM_CLASSES
class_total = [0] * NUM_CLASSES
with torch.no_grad():
    for sequences, labels in val_dataloader:
        sequences, labels = sequences.cuda(), labels.cuda()
        outputs = model(sequences.unsqueeze(1))
        _, predicted = torch.max(outputs, 1)  # Get the predicted class index
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i].item()  # Get the actual label as an integer
            class_correct[label] += c[i].item()  # Increment correct count if prediction was right
            class_total[label] += 1  # Always increment total count

# Print accuracy for each class
for i in range(NUM_CLASSES):
    if class_total[i] > 0:
        print(f'Accuracy of class {i}: {100 * class_correct[i] / class_total[i]:.2f}%')
    else:
        print(f'Accuracy of class {i}: N/A (no samples)')

