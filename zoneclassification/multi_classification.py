import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Data normalization
def min_max_normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)

# Custom dataset class
class FinancialTimeSeriesDataset(Dataset):
    def __init__(self, dataframe, window_length=16, indexes=None):
        self.dataframe = dataframe
        self.window_length = window_length
        self.indexes = indexes if indexes is not None else range(len(dataframe))
        self.samples, self.labels = self.create_samples_and_labels()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def create_samples_and_labels(self):
        samples = []
        labels = []
        for idx in self.indexes:
            current_candle = self.dataframe.iloc[idx]
            prev_candles = self.dataframe.iloc[idx - self.window_length:idx]
            next_candles = self.dataframe.iloc[idx + 1:idx + 1 + self.window_length]

            start_idx = idx - self.window_length
            end_idx = idx + 1
            sample = self.dataframe.iloc[start_idx:end_idx].values.astype(np.float32)

            if current_candle['Low'] < prev_candles['Low'].min() and current_candle['Low'] < next_candles['Low'].min() and (
                    next_candles['High'] >= current_candle['Low'] * 1.01).any():
                label = 0.0
            elif current_candle['High'] > prev_candles['High'].max() and current_candle['High'] > next_candles['High'].max() and (
                    next_candles['Low'] <= current_candle['High'] * 0.99).any():
                label = 1.0
            else:
                label = 2.0

            samples.append(sample)
            labels.append(label)

        samples = np.array(samples)
        labels = np.array(labels)

        return samples, labels

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size)  # Ensuring the output size is 16 (embedding size)

    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)

        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Prototypical Networks Specific Methods
def compute_prototypes(features, labels, num_classes):
    prototypes = []
    for cls in range(num_classes):
        class_features = features[labels == cls]
        if len(class_features) > 0:
            prototype = class_features.mean(dim=0)
        else:
            # Handle cases with no samples in a class
            prototype = torch.zeros(features.size(1)).to(features.device)
        prototypes.append(prototype)
    return torch.stack(prototypes)

def euclidean_distance(a, b):
    return ((a - b) ** 2).sum(dim=-1)

def prototypical_loss(prototypes, features, labels):
    distances = euclidean_distance(prototypes.unsqueeze(0), features.unsqueeze(1))
    # Prevent underflow/overflow by using logsumexp trick
    log_p_y = -distances
    log_p_y = log_p_y - torch.logsumexp(log_p_y, dim=1, keepdim=True)
    return torch.nn.functional.nll_loss(log_p_y, labels)

# Load data
all_samples = pd.read_csv('data/BTCUSD_Hourly_Ask_2022.01.01_2024.07.01.csv')
all_samples.drop(['Time (EET)'], inplace=True, axis=1)

# Normalizing data
all_samples = all_samples.apply(min_max_normalize)

window_length = 8
total_sequences = all_samples.shape[0]

train_size = int(total_sequences * 0.8)
val_size = int(total_sequences * 0.1)
test_size = total_sequences - train_size - val_size

indices = list(range(32, len(all_samples) - 2 * window_length))

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

train_dataset = FinancialTimeSeriesDataset(all_samples, window_length, train_indices)
val_dataset = FinancialTimeSeriesDataset(all_samples, window_length, val_indices)
test_dataset = FinancialTimeSeriesDataset(all_samples, window_length, test_indices)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = all_samples.shape[1]
hidden_size = 8  # Set embedding size to 16
num_classes = 3
num_layers = 2
num_epochs = 100
learning_rate = 0.001
patience = 10

model = LSTMModel(input_size, hidden_size, num_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with gradient clipping and checks
best_val_loss = float('inf')
epochs_no_improve = 0
clip_value = 1.0  # Value for clipping gradients

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for samples, labels in train_dataloader:
        samples, labels = samples.to(device), labels.to(device)

        # Split the batch into support and query sets
        support_size = samples.size(0) // 2
        support_samples = samples[:support_size]
        support_labels = labels[:support_size]
        query_samples = samples[support_size:]
        query_labels = labels[support_size:]

        # Compute the embeddings for support and query samples
        support_embeddings = model(support_samples)
        query_embeddings = model(query_samples)

        # Compute prototypes using support embeddings
        prototypes = compute_prototypes(support_embeddings, support_labels, num_classes)

        # Compute the loss using query embeddings and prototypes
        loss = prototypical_loss(prototypes, query_embeddings, query_labels)

        if torch.isnan(loss):
            print(f"NaN loss detected at epoch {epoch}, batch {samples}")
            continue

        optimizer.zero_grad()
        loss.backward()

        # Clip gradients to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_dataloader)

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for samples, labels in val_dataloader:
            samples, labels = samples.to(device), labels.to(device)

            # Split into support and query sets
            support_size = samples.size(0) // 2
            support_samples = samples[:support_size]
            support_labels = labels[:support_size]
            query_samples = samples[support_size:]
            query_labels = labels[support_size:]

            # Compute embeddings and prototypes
            support_embeddings = model(support_samples)
            query_embeddings = model(query_samples)
            prototypes = compute_prototypes(support_embeddings, support_labels, num_classes)

            # Compute loss
            loss = prototypical_loss(prototypes, query_embeddings, query_labels)
            val_loss += loss.item()

    val_loss /= len(val_dataloader)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Save the model for this epoch
    torch.save(model.state_dict(), f'saved_models/model_epoch_{epoch + 1}.pth')

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'saved_models/best_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print("Early stopping!")
            break

# Load the best model
model.load_state_dict(torch.load('saved_models/best_model.pth'))

# Test accuracy
model.eval()
class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))

with torch.no_grad():
    for sequences, labels in test_dataloader:
        sequences, labels = sequences.to(device), labels.to(device)
        embeddings = model(sequences)
        prototypes = compute_prototypes(embeddings, labels, num_classes)
        distances = euclidean_distance(prototypes.unsqueeze(0), embeddings.unsqueeze(1))
        _, predicted = torch.min(distances, dim=1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i].item()
            class_correct[label] += c[i].item()
            class_total[label] += 1

# Print accuracy for each class
for i in range(num_classes):
    if class_total[i] > 0:
        print(f'Accuracy of class {i}: {100 * class_correct[i] / class_total[i]:.2f}%')
    else:
        print(f'Accuracy of class {i}: N/A (no samples)')
