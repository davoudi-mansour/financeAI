import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Data normalization
def min_max_normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)

# RSI calculation
def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# MACD calculation
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line

# SMA calculation
def calculate_sma(data, window=30):
    sma = data['Close'].rolling(window=window).mean()
    return sma

# EMA calculation
def calculate_ema(data, window=30):
    ema = data['Close'].ewm(span=window, adjust=False).mean()
    return ema

# Bollinger Bands calculation
def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = calculate_sma(data, window)
    rolling_std = data['Close'].rolling(window=window).std()
    upper_band = sma + (rolling_std * num_std)
    lower_band = sma - (rolling_std * num_std)
    return upper_band, lower_band

# ATR calculation
def calculate_atr(data, window=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    tr = high_low.combine(high_close, max).combine(low_close, max)
    atr = tr.rolling(window=window).mean()
    return atr

# ADX calculation
def calculate_adx(data, window=14):
    plus_dm = data['High'].diff()
    minus_dm = data['Low'].diff()
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
    minus_dm = -minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)

    tr = calculate_atr(data, window)

    plus_di = 100 * (plus_dm.ewm(span=window, adjust=False).mean() / tr)
    minus_di = 100 * (minus_dm.ewm(span=window, adjust=False).mean() / tr)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=window).mean()
    return adx

# Stochastic Oscillator calculation
def calculate_stochastic_oscillator(data, window=14):
    low_min = data['Low'].rolling(window=window).min()
    high_max = data['High'].rolling(window=window).max()
    stoch = 100 * (data['Close'] - low_min) / (high_max - low_min)
    return stoch

# Williams %R calculation
def calculate_williams_r(data, window=14):
    high_max = data['High'].rolling(window=window).max()
    low_min = data['Low'].rolling(window=window).min()
    will_r = -100 * (high_max - data['Close']) / (high_max - low_min)
    return will_r

# CCI calculation
def calculate_cci(data, window=20):
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    cci = (tp - tp.rolling(window=window).mean()) / (0.015 * tp.rolling(window=window).std())
    return cci

# OBV calculation
def calculate_obv(data):
    obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    return obv

# MFI calculation
def calculate_mfi(data, window=14):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    money_flow = typical_price * data['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
    mfi = 100 - (100 / (1 + (positive_flow.rolling(window=window).sum() /
                             negative_flow.rolling(window=window).sum())))
    return mfi

# Adding technical indicators
def add_technical_indicators(data):
    data['RSI'] = calculate_rsi(data)
    data['MACD'], data['Signal_Line'] = calculate_macd(data)
    data['SMA'] = calculate_sma(data)
    data['EMA'] = calculate_ema(data)
    data['Upper_Band'], data['Lower_Band'] = calculate_bollinger_bands(data)
    data['ATR'] = calculate_atr(data)
    data['ADX'] = calculate_adx(data)
    data['Stochastic'] = calculate_stochastic_oscillator(data)
    data['Williams_R'] = calculate_williams_r(data)
    data['CCI'] = calculate_cci(data)
    data['OBV'] = calculate_obv(data)
    data['MFI'] = calculate_mfi(data)
    data.dropna(inplace=True)  # Remove rows with NaN values created by indicators
    return data

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
                label = 1.0
            elif current_candle['High'] > prev_candles['High'].max() and current_candle['High'] > next_candles['High'].max() and (
                next_candles['Low'] <= current_candle['High'] * 0.99).any():
                label = 1.0
            else:
                label = 0.0

            samples.append(sample)
            labels.append(label)

        samples = np.array(samples)
        labels = np.array(labels)
        return samples, labels

def get_dataloaders(file_path, window_length=16, batch_size=128):
    # Load data
    all_samples = pd.read_csv(file_path)
    all_samples.drop(['Time (EET)'], inplace=True, axis=1)

    # Adding indicators and normalizing data
    all_samples = add_technical_indicators(all_samples)
    all_samples = all_samples.apply(min_max_normalize)

    total_sequences = all_samples.shape[0]
    train_size = int(total_sequences * 0.8)
    val_size = int(total_sequences * 0.1)

    indices = list(range(32, len(all_samples) - 2 * window_length))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_dataset = FinancialTimeSeriesDataset(all_samples, window_length, train_indices)
    val_dataset = FinancialTimeSeriesDataset(all_samples, window_length, val_indices)
    test_dataset = FinancialTimeSeriesDataset(all_samples, window_length, test_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
