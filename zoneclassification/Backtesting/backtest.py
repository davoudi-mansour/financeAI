from dataset import get_dataloaders
import yaml
import torch
from autoencoder_model import LSTMAutoencoder
from lstm_model import LSTMClassifier
import pandas as pd
import plotly.graph_objects as go

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_model(config, device):
    # Hyperparameters
    input_size = config['model_params']['input_size']
    hidden_size = config['model_params']['hidden_size']
    num_layers = config['model_params']['num_layers']
    batch_size = config['testing_params']['batch_size']
    threshold = float(config['testing_params']['threshold'])
    window_length = 16

    test_dataloader = get_dataloaders(config['data']['data_path'], window_length, batch_size)
    # Initialize models
    autoencoder_model = LSTMAutoencoder(input_size, hidden_size, num_layers).to(device)
    classifier_model = LSTMClassifier(input_size, hidden_size, num_layers).to(device)

    # Load the best model
    autoencoder_model.load_state_dict(torch.load('saved_models/best_autoencoder.pth'))
    classifier_model.load_state_dict(torch.load('saved_models/best_classifier.pth'))
    autoencoder_model.eval()
    classifier_model.eval()
    predictions = []
    all_counter = 0
    buy_counter = 0
    sell_counter = 0
    with torch.no_grad():
        for samples, _ in test_dataloader:
            all_counter += 1
            samples = samples.to(device)
            reconstructed, _ = autoencoder_model(samples)
            reconstruction_error = torch.mean((reconstructed - samples) ** 2, dim=[1, 2]).cpu().item()
            if reconstruction_error > 1.5 * threshold:
                classifier_output = classifier_model(samples).cpu().item()
                if classifier_output > 0.99:
                    buy_counter += 1
                    predictions.append(1)
                elif classifier_output < 0.01:
                    sell_counter += 1
                    predictions.append(-1)
                else:
                    predictions.append(0)
            else:
                predictions.append(0)

    return predictions

def prepare_backtesting_data(config, predictions):
    data_path = config['data']['data_path']
    # Load data
    all_samples = pd.read_csv(data_path)
    window_length = 16
    total_sequences = all_samples.shape[0]
    train_size = int(total_sequences * 0.8)
    val_size = int(total_sequences * 0.1)
    indices = list(range(32, len(all_samples) - 2 * window_length))
    test_indices = indices[train_size + val_size:]
    test_data = all_samples.iloc[test_indices].copy()
    test_data.loc[:, 'signal'] = predictions
    return test_data

if __name__ == "__main__":
    config_path = 'config.yaml'
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictions = test_model(config, device)
    df = prepare_backtesting_data(config, predictions)
    # Convert 'Time (EET)' to datetime
    df['Time (EET)'] = pd.to_datetime(df['Time (EET)'], format='%Y.%m.%d %H:%M:%S')

    # Create the candlestick chart
    fig = go.Figure(data=[go.Candlestick(x=df['Time (EET)'],
                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'],
                                         name='Candlesticks')])

    # Add blue arrows for buy signals (signal == 1) on the low of the candle
    buy_signals = df[df['signal'] == 1]
    fig.add_trace(go.Scatter(x=buy_signals['Time (EET)'],
                             y=buy_signals['Low'],
                             mode='text',
                             text=['\u2191'] * len(buy_signals),
                             textfont=dict(color='#3a7ca5', size=30),  # Increase the font size for thicker arrows
                             name='Buy Signal'))

    # Add yellow arrows for sell signals (signal == -1) on the high of the candle
    sell_signals = df[df['signal'] == -1]
    fig.add_trace(go.Scatter(x=sell_signals['Time (EET)'],
                             y=sell_signals['High'],
                             mode='text',
                             text=['\u2193'] * len(sell_signals),
                             textfont=dict(color='#fee440', size=30),  # Increase the font size for thicker arrows
                             name='Sell Signal'))

    # Update layout
    fig.update_layout(title='Bitcoin Candlestick Chart with Buy/Sell Signals',
                      xaxis_title='Time (EET)',
                      yaxis_title='Price (USD)',
                      xaxis_rangeslider_visible=False)

    # Show the plot
    fig.show()
