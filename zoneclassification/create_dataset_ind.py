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

def calculate_rsi(prices, window=14):
    deltas = np.diff(prices)
    seed = deltas[:window+1]
    up = seed[seed >= 0].sum() / window
    down = -seed[seed < 0].sum() / window
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:window] = 100. - 100. / (1. + rs)
    for i in range(window, len(prices)):
        delta = deltas[i - 1]  # Change in price at i
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        # Calculate average gain and loss over the window
        up = (up * (window - 1) + upval) / window
        down = (down * (window - 1) + downval) / window
        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)
    return rsi

def calculate_ema(data, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()  # Normalize weights
    ema = np.convolve(data, weights, mode='full')[:len(data)]
    ema[:window] = ema[window]
    return ema

def calculate_macd(prices, short_window=12, long_window=26, signal_window=9):
    short_ema = calculate_ema(prices, short_window)
    long_ema = calculate_ema(prices, long_window)
    macd_line = short_ema - long_ema
    signal_line = calculate_ema(macd_line, signal_window)
    return macd_line, signal_line, short_ema, long_ema

_15min_samples = np.load('data/_5min_samples.npy')

labels = []
zone_area = 0.003
window_length = 10000
pivot_index = window_length
sequences = []

for i in range(0, len(_15min_samples)-window_length):
    sequence = _15min_samples[pivot_index-window_length:pivot_index]
    last_price = sequence[-1]
    counter = 0
    if pivot_index < len(_15min_samples)-1:
        while True:
            price_change = _15min_samples[pivot_index] - last_price
            counter += 1
            if price_change >= zone_area:
                label = 0
                normalized_sequence = min_max_normalize(sequence)
                #rsi_sequence = calculate_rsi(sequence)
                #macd_line, signal_line, short_ema, long_ema = calculate_macd(sequence)
                # data = np.concatenate(
                #     (sequence, rsi_sequence))
                sequences.append(normalized_sequence)
                labels.append(label)
                break

            elif price_change <= -zone_area:
                label = 1
                # fft = np.fft.fft(np.asarray(sequence))
                # num_ = 3
                # fft_list_m10 = np.copy(fft)
                # fft_list_m10[num_:-num_] = 0
                # fourier_sequence = np.fft.ifft(fft_list_m10)
                normalized_sequence = min_max_normalize(sequence)
                # rsi_sequence = calculate_rsi(sequence)
                # macd_line, signal_line, short_ema, long_ema = calculate_macd(sequence)
                # data = np.concatenate(
                #     (sequence, rsi_sequence))
                sequences.append(normalized_sequence)
                labels.append(label)
                break
            if pivot_index < (len(_15min_samples) - window_length - 1):
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
plt.savefig('results/_15min_distribution.png')

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