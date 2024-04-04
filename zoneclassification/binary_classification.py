import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_weights = nn.Parameter(torch.Tensor(hidden_size, 1))
        nn.init.uniform_(self.attention_weights, -0.1, 0.1)

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, num_directions * hidden_size)
        energy = torch.matmul(lstm_output, self.attention_weights)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(energy.squeeze(-1), dim=1)  # (batch_size, seq_len)
        # Calculate the weighted sum of LSTM output
        context_vector = torch.matmul(lstm_output.permute(0, 2, 1), attention_weights.unsqueeze(-1)).squeeze(-1)
        return context_vector

class CNNBiLSTMWithAttention(nn.Module):
    def __init__(self, dropout_rate=0.5, l2_reg=0.001):
        super(CNNBiLSTMWithAttention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5)
        self.batch_norm = nn.BatchNorm1d(32)  # Batch normalization layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.lstm = nn.LSTM(input_size=32, hidden_size=32, num_layers=2, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size=64)
        self.fc = nn.Linear(64, 1)  # Output size 1 for binary classification
        self.sigmoid = nn.Sigmoid()

        # Add L2 regularization to the linear layer
        self.fc.weight_regularizer = torch.nn.Parameter(torch.randn(1, 64) * l2_reg)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)  # Apply batch normalization
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = x.permute(0, 2, 1)  # Reshape for LSTM input
        output, _ = self.lstm(x)
        # Apply attention mechanism
        context_vector = self.attention(output)
        x = self.fc(context_vector)
        x = self.sigmoid(x)  # Apply sigmoid activation
        return x

class CNNBiLSTM(nn.Module):
    def __init__(self, dropout_rate=0.2, l2_reg=0.001):
        super(CNNBiLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=2)
        self.batch_norm = nn.BatchNorm1d(64)  # Batch normalization layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.lstm = nn.LSTM(input_size=64, hidden_size=16, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(32, 1)  # Output size 1 for binary classification
        self.sigmoid = nn.Sigmoid()

        # Add L2 regularization to the linear layer
        self.fc.weight_regularizer = torch.nn.Parameter(torch.randn(64, 1) * l2_reg)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)  # Apply batch normalization
        x = self.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # Reshape for LSTM input
        output, _ = self.lstm(x)
        x = output[:, -1, :]  # Use only the last output of LSTM
        x = self.fc(x)
        x = self.sigmoid(x)  # Apply sigmoid activation
        return x

class CNNTransformer(nn.Module):
    def __init__(self, d_model=16, nhead=2, num_layers=1, dropout_rate=0.5, l2_reg=0.001):
        super(CNNTransformer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=3)
        self.batch_norm = nn.BatchNorm1d(d_model)  # Batch normalization layer
        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.transformer_encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)  # Output size 1 for binary classification
        self.sigmoid = nn.Sigmoid()

        # Add L2 regularization to the linear layer
        self.fc.weight_regularizer = torch.nn.Parameter(torch.randn(1, d_model) * l2_reg)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)  # Apply batch normalization
        x = self.dropout(x)  # Apply dropout
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # Reshape for Transformer input
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)  # Global average pooling
        x = self.fc(x)
        x = self.sigmoid(x)  # Apply sigmoid activation
        return x

# Initialize model, loss function, and optimizer

model = CNNTransformer().cuda()  # Move model to GPU
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load DataLoader
with open('dataset/hourly_dataloader.pkl', 'rb') as f:
    loaded_dataloader = pickle.load(f)

# Generate indices for train-validation split
indices = list(range(len(loaded_dataloader.dataset)))
train_indices, val_indices = train_test_split(indices, test_size=0.1, random_state=42)

# Create Subset objects for training, validation, and testing sets
train_dataset = Subset(loaded_dataloader.dataset, train_indices)
val_dataset = Subset(loaded_dataloader.dataset, val_indices)

# Create DataLoader for training, validation, and testing sets with batch size 32
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

# Early stopping parameters
patience = 5
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
        sequences, labels = sequences.cuda(), labels.cuda()  # Move data to GPU
        optimizer.zero_grad()
        outputs = model(sequences.unsqueeze(1))  # Add channel dimension
        loss = criterion(outputs.squeeze(), labels.float())  # Squeeze to remove extra dimension, convert labels to float
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
            val_loss += criterion(val_outputs.squeeze(), val_labels.float()).item()
    val_loss /= len(val_dataloader)
    val_losses.append(val_loss)
    print(f"Validation Loss: {val_loss}")

    # Check for early stopping and save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        if best_model_path:
            os.remove(best_model_path)  # Remove the previous best model
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
plt.savefig('loss.png')  # Save the plot as PNG image
plt.show()

# Load the best model state
if best_model_path:
    model.load_state_dict(torch.load(best_model_path))

# Evaluation loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for sequences, labels in val_dataloader:
        sequences, labels = sequences.cuda(), labels.cuda()  # Move data to GPU
        outputs = model(sequences.unsqueeze(1))  # Add channel dimension
        predicted = (outputs > 0.5).float()  # Apply threshold 0.5 for binary classification
        total += labels.size(0)
        correct += (predicted.squeeze() == labels.float()).sum().item()  # Squeeze outputs to match labels dimension

print(f"Accuracy on test set: {100 * correct / total}%")
