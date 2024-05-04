import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pickle

def min_max_normalize(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data

_15min_samples = np.load('data/_5min_samples.npy')

_15min_samples = _15min_samples[2000000:]
labels = []
zone_area = 0.0037
stop_area = zone_area / 2
window_length = 10000
pivot_index = window_length
sequences = []

def validate(stop_area, last_price, val_sequence, state):
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

for i in range(0, len(_15min_samples)-window_length, 36):
    sequence = _15min_samples[i:i+window_length]
    last_price = sequence[-1]
    candidate = i+window_length
    print(i)
    while True:
        if candidate>=len(_15min_samples):
            break
        price_change = _15min_samples[candidate] - last_price
        if price_change >= zone_area:
            val_sequence = _15min_samples[i+window_length:candidate]
            if validate(stop_area, last_price, val_sequence, state=0) == False:
                label = 2
                normalized_sequence = min_max_normalize(sequence)
                sequences.append(normalized_sequence)
                labels.append(label)
                break

            else:
                label = 0
                normalized_sequence = min_max_normalize(sequence)
                sequences.append(normalized_sequence)
                labels.append(label)
                break
        elif price_change <= -zone_area:
            val_sequence = _15min_samples[i + window_length:candidate]

            if validate(stop_area, last_price, val_sequence, state=1) == False:
                label = 2
                normalized_sequence = min_max_normalize(sequence)
                sequences.append(normalized_sequence)
                labels.append(label)
                break
            else:
                label = 1
                normalized_sequence = min_max_normalize(sequence)
                sequences.append(normalized_sequence)
                labels.append(label)
                break
        candidate += 1

label_counts = np.unique(labels, return_counts=True)
# Plot the distribution
plt.bar(label_counts[0], label_counts[1])
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.title('Label Distribution')
plt.savefig('results/_5min_distribution.png')

sequences = torch.tensor(np.array(sequences), dtype=torch.float32)
labels = torch.tensor(np.array(labels), dtype=torch.long)
# Create DataLoader
dataset = TensorDataset(sequences, labels)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Save DataLoader
with open('dataset/_5min_dataloader.pkl', 'wb') as f:
    pickle.dump(dataloader, f)

# Load DataLoader
with open('dataset/_5min_dataloader.pkl', 'rb') as f:
    loaded_dataloader = pickle.load(f)

# Example of iterating through loaded DataLoader
for sequence, label in loaded_dataloader:
    print("Sequence:", sequence)
    print("Label:", label)