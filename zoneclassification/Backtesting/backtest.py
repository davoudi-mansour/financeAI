import backtrader as bt
from dataset import get_dataloaders
import yaml
import torch
from autoencoder_model import LSTMAutoencoder
from lstm_model import LSTMClassifier
import pandas as pd
import numpy as np

class CustomStrategy(bt.SignalStrategy):
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.last_action = None
        self.buy_price = None

    def next(self):
        current_signal = self.data.signal[0]

        if current_signal == 1 and (self.last_action is None or self.last_action == 'sell'):
            self.buy()
            self.last_action = 'buy'
            self.buy_price = self.dataclose[0]
        elif current_signal == -1 and self.last_action == 'buy' and self.dataclose[0] > self.buy_price:
            self.sell()
            self.last_action = 'sell'
            self.buy_price = None

class CustomPandasData(bt.feeds.PandasData):
    lines = ('signal',)
    params = (
        ('signal', -1),
    )

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_model(config, device):
    # Hyperparameters
    input_size = config['model_params']['input_size']
    hidden_size = config['model_params']['hidden_size']
    num_layers = config['model_params']['num_layers']
    batch_size = config['testing_params']['batch_size']
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
    threshold = 0.0005762064
    all_counter = 0
    buy_counter = 0
    sell_counter = 0
    with torch.no_grad():
        for samples, _ in test_dataloader:
            all_counter += 1
            samples = samples.to(device)
            reconstructed, _ = autoencoder_model(samples)
            reconstruction_error = torch.mean((reconstructed - samples) ** 2, dim=[1, 2]).cpu().item()
            if reconstruction_error > 1.8 * threshold:
                classifier_output = classifier_model(samples).cpu().item()
                if classifier_output > 0.90:
                    buy_counter += 1
                    predictions.append(1)
                elif classifier_output < 0.10:
                    sell_counter += 1
                    predictions.append(-1)
                else:
                    predictions.append(0)
            else:
                predictions.append(0)
        print(all_counter)
        print(buy_counter)
        print(sell_counter)
    return predictions

def prepare_backtrader_data(config, predictions):
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
    #test_data.set_index('Time (EET)', inplace=True)
    #test_data.index = pd.to_datetime(test_data.index, format='%Y.%m.%d %H:%M:%S')
    test_data.to_csv('output_result.csv', index=False)
    exit()
    return test_data

if __name__ == "__main__":
    config_path = 'config.yaml'
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictions = test_model(config, device)
    data = prepare_backtrader_data(config, predictions)

    # Convert the data into a format compatible with Backtrader
    data_feed = CustomPandasData(dataname=data)

    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add the strategy
    cerebro.addstrategy(CustomStrategy)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data_feed)

    # Set our desired cash start
    cerebro.broker.setcash(10000.0)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)

    # Set the commission
    cerebro.broker.setcommission(commission=0.001)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Plot the result
    cerebro.plot()
