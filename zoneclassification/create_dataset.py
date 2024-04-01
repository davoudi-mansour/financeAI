import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pickle

hourly_samples = np.load('data/hourly_samples.npy')
sequences = []
labels = []
zone_area = 0.3
window_length = 5000
pivot_index = window_length

for i in range(0, len(hourly_samples)-window_length):

    sequence = hourly_samples[pivot_index-window_length:pivot_index]
    last_price = sequence[-1]
    counter = 0
    if pivot_index < len(hourly_samples)-1:
        while True:
            price_change = hourly_samples[pivot_index] - last_price
            counter += 1
            if price_change >= zone_area:
                label = 1
                normalized_sequence = (np.array(sequence) - np.min(sequence)) / (np.max(sequence) - np.min(sequence))
                sequences.append(normalized_sequence)
                labels.append(label)
                break
            elif price_change < -zone_area:
                label = 0
                normalized_sequence = (np.array(sequence) - np.min(sequence)) / (np.max(sequence) - np.min(sequence))
                sequences.append(normalized_sequence)
                labels.append(label)
                break
            if pivot_index < (len(hourly_samples) - window_length - 1):
                pivot_index += 1
            else:
                break
        pivot_index += counter

label_counts = np.unique(labels, return_counts=True)
# Plot the distribution
plt.bar(label_counts[0], label_counts[1])
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.title('Label Distribution')
plt.savefig('results/hourly_distribution.png')

sequences = torch.tensor(np.array(sequences), dtype=torch.float32)
labels = torch.tensor(np.array(labels), dtype=torch.long)
# Create DataLoader
dataset = TensorDataset(sequences, labels)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Save DataLoader
with open('dataset/hourly_dataloader.pkl', 'wb') as f:
    pickle.dump(dataloader, f)

# Load DataLoader
with open('dataset/hourly_dataloader.pkl', 'rb') as f:
    loaded_dataloader = pickle.load(f)

# Example of iterating through loaded DataLoader
for sequence, label in loaded_dataloader:
    print("Sequence:", sequence)
    print("Label:", label)