import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample


def min_max_normalize(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]


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
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def create_samples_and_labels(self):
        samples = []
        labels = []
        for idx in self.indexes:
            current_candle = self.dataframe.iloc[idx]
            prev_candles = self.dataframe.iloc[idx - self.window_length:idx]
            next_candles = self.dataframe.iloc[idx + 1:idx + 1 + self.window_length]

            start_idx = idx - self.window_length
            end_idx = idx + self.window_length + 1
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

        # Resample to balance the dataset
        samples, labels = self.resample_classes(samples, labels, classes=[0.0, 1.0])

        return samples, labels

    def resample_classes(self, samples, labels, classes):
        resampled_samples = []
        resampled_labels = []

        for cls in classes:
            class_samples = samples[labels == cls]
            class_labels = labels[labels == cls]
            resampled_class_samples, resampled_class_labels = resample(
                class_samples,
                class_labels,
                replace=True,
                n_samples=len(samples[labels == 2.0]),  # Assuming class 2 is the majority class
                random_state=42
            )
            resampled_samples.append(resampled_class_samples)
            resampled_labels.append(resampled_class_labels)

        # Combine resampled classes with the original majority class
        majority_samples = samples[labels == 2.0]
        majority_labels = labels[labels == 2.0]

        resampled_samples.append(majority_samples)
        resampled_labels.append(majority_labels)

        resampled_samples = np.vstack(resampled_samples)
        resampled_labels = np.hstack(resampled_labels)

        return resampled_samples, resampled_labels


def draw_candle_sample(dataloader, idx):
    dataset = dataloader.dataset
    sample, label = dataset[idx]

    # Convert to DataFrame for Plotly
    sample_df = pd.DataFrame(sample.numpy(), columns=dataset.dataframe.columns)

    # Plot with Plotly
    fig = go.Figure(data=[go.Candlestick(x=sample_df.index,
                                         open=sample_df['Open'],
                                         high=sample_df['High'],
                                         low=sample_df['Low'],
                                         close=sample_df['Close'])])
    fig.update_layout(title=f'Candlestick Chart - Label: {label.item()}',
                      xaxis_title='Index',
                      yaxis_title='Price')
    fig.show()

    return label.item()


def draw_label_distribution(dataloader):
    labels = []
    for sample, label in dataloader:
        labels.extend(label.numpy())
    labels = np.array(labels)

    label_counts = np.bincount(labels.astype(int))

    fig = go.Figure(data=[go.Bar(x=[0, 1, 2], y=label_counts)])
    fig.update_layout(title='Distribution of Labels',
                      xaxis_title='Label',
                      yaxis_title='Count')
    fig.show()


def get_indices_with_label_zero(dataloader):
    indices_with_label_zero = []
    for idx, (sample, label) in enumerate(dataloader.dataset):
        if label.item() == 0.0:
            indices_with_label_zero.append(dataloader.dataset.indexes[idx])
    return indices_with_label_zero


# Load data
all_samples = pd.read_csv('data/BTCUSD_Hourly_Ask_2022.01.01_2024.07.01.csv')
all_samples.drop(['Time (EET)'], inplace=True, axis=1)

window_length = 16
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

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Example usage
idx_to_draw = 22  # Change this index as needed
label = draw_candle_sample(train_dataloader, idx_to_draw)
print(f'Label of the drawn sample: {label}')

# Draw the distribution of labels
draw_label_distribution(train_dataloader)

# Get indices with label zero
indices_with_label_zero = get_indices_with_label_zero(train_dataloader)
print(f'Indices with label zero: {indices_with_label_zero}')