import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt

NUM_CLASSES = 4

class CNNBiLSTM(nn.Module):
    def __init__(self, dropout_rate=0.1, l2_reg=0.001, num_classes=NUM_CLASSES):
        super(CNNBiLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5)
        self.batch_norm = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.lstm = nn.LSTM(input_size=32, hidden_size=16, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(32, num_classes)  # Output size changed to num_classes for multi-class classification

        # Add L2 regularization to the linear layer
        self.fc.weight_regularizer = torch.nn.Parameter(torch.randn(64, num_classes) * l2_reg)

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
plt.savefig('result.png')
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
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# Print accuracy for each class
for i in range(NUM_CLASSES):
    if class_total[i] > 0:
        print(f'Accuracy of class {i + 1}: {100 * class_correct[i] / class_total[i]}%')
    else:
        print(f'Accuracy of class {i + 1}: N/A (no samples)')