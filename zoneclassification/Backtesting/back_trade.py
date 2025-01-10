import backtrader as bt
import pandas as pd
import yaml
import torch
from dataset import get_dataloaders
from autoencoder_model import LSTMAutoencoder
from lstm_model import LSTMClassifier

# Load configuration function from your original code
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Prediction and data preparation functions from your code
def test_model(config, device):
    input_size = config['model_params']['input_size']
    hidden_size = config['model_params']['hidden_size']
    num_layers = config['model_params']['num_layers']
    batch_size = config['testing_params']['batch_size']
    threshold = float(config['testing_params']['threshold'])
    window_length = 16

    test_dataloader = get_dataloaders(config['data']['data_path'], window_length, batch_size)
    autoencoder_model = LSTMAutoencoder(input_size, hidden_size, num_layers).to(device)
    classifier_model = LSTMClassifier(input_size, hidden_size, num_layers).to(device)

    autoencoder_model.load_state_dict(torch.load('saved_models/best_autoencoder.pth'))
    classifier_model.load_state_dict(torch.load('saved_models/best_classifier.pth'))
    autoencoder_model.eval()
    classifier_model.eval()
    predictions = []

    with torch.no_grad():
        for samples, _ in test_dataloader:
            samples = samples.to(device)
            reconstructed, _ = autoencoder_model(samples)
            reconstruction_error = torch.mean((reconstructed - samples) ** 2, dim=[1, 2]).cpu().item()
            if reconstruction_error > 5 * threshold:
                classifier_output = classifier_model(samples).cpu().item()
                if classifier_output > 0.9998:
                    predictions.append(1)
                elif classifier_output < 0.0001:
                    predictions.append(-1)
                else:
                    predictions.append(0)
            else:
                predictions.append(0)

    return predictions

def prepare_backtesting_data(config, predictions):
    data_path = config['data']['data_path']
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

# Define a Backtrader data feed that includes the 'signal' column
class PandasSignalData(bt.feeds.PandasData):
    # Add 'signal' as a line to the data feed
    lines = ('signal',)
    params = (
        ('signal', -1),  # default -1 indicates no signal
    )

# Create a Backtrader Strategy that uses the 'signal' column
class SignalStrategy(bt.Strategy):
    params = (
        ('stop_loss', 0.1),  # Stop loss percentage for buy orders (5%)
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.data_signal = self.datas[0].signal
        self.order = None  # To keep track of open orders
        self.buy_price = None  # Track the price of the last buy order
        self.stop_price = None  # Track the stop-loss price for the buy order
        self.successful_trades = 0  # Counter for successful trades
        self.failed_trades = 0  # Counter for failed trades
        self.current_trade_result = None  # Track the current trade result

    def next(self):
        # Skip if there's an open order
        if self.order:
            return

        # Buy signal
        if self.data_signal[0] == 1 and not self.position:
            # Place a buy order
            self.buy_price = self.dataclose[0]
            self.stop_price = self.buy_price * (1 - self.params.stop_loss)
            self.order = self.buy(size=1)  # Adjust size as per your requirements
            print(f"Buy at {self.buy_price}, Stop-Loss set at {self.stop_price}")

        # Check for stop-loss on a buy position
        elif self.position and self.buy_price and self.dataclose[0] < self.stop_price:
            # Sell to close the buy position if stop-loss is triggered
            self.order = self.sell(size=self.position.size)
            self.current_trade_result = -1  # Mark the trade as failed
            print(f"Stop-Loss triggered. Sell at {self.dataclose[0]}")

        # Sell signal handling
        elif self.data_signal[0] == -1 and self.position:
            # Only act on the sell signal if the current price is above the buy price
            # This avoids premature selling at a loss unless stop-loss is hit
            if self.dataclose[0] > self.buy_price:
                # Sell to close the position
                self.order = self.sell(size=self.position.size)
                if self.dataclose[0] > self.buy_price:
                    self.current_trade_result = 1  # Mark the trade as successful
                else:
                    self.current_trade_result = -1  # Mark the trade as failed
                print(f"Sell at {self.dataclose[0]}")
            else:
                # If price is lower than buy price, ignore the sell signal
                print(f"Sell signal ignored at {self.dataclose[0]} since price is below buy price.")

    def notify_order(self, order):
        """
        Handle order notifications and reset tracking when orders are completed.
        """
        if order.status in [order.Completed]:
            # Update counters based on the trade result
            if self.current_trade_result == 1:
                self.successful_trades += 1
            elif self.current_trade_result == -1:
                self.failed_trades += 1

            # Reset trade tracking variables
            self.order = None
            self.current_trade_result = None

    def stop(self):
        """
        At the end of the backtest, print the trade success/failure ratio.
        """
        total_trades = self.successful_trades + self.failed_trades
        if total_trades > 0:
            success_ratio = self.successful_trades / total_trades
            print(f"Successful Trades: {self.successful_trades}")
            print(f"Failed Trades: {self.failed_trades}")
            print(f"Success Ratio: {success_ratio:.2f}")
        else:
            print("No trades were executed.")

if __name__ == "__main__":
    # Load configuration and device
    config_path = 'config.yaml'
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate predictions and prepare backtesting data
    predictions = test_model(config, device)
    df = prepare_backtesting_data(config, predictions)

    # Convert time column to datetime if necessary
    df['Time (EET)'] = pd.to_datetime(df['Time (EET)'], format='%Y.%m.%d %H:%M:%S')
    df.set_index('Time (EET)', inplace=True)

    # Initialize Cerebro engine
    cerebro = bt.Cerebro()

    # Add strategy to Cerebro
    cerebro.addstrategy(SignalStrategy)

    # Create a data feed from the DataFrame including the 'signal' column
    data = PandasSignalData(
        dataname=df,
        datetime=None,     # using index as datetime
        open='Open',
        high='High',
        low='Low',
        close='Close',
        volume='Volume' if 'Volume' in df.columns else None,
        signal='signal'
    )

    # Add data feed to Cerebro
    cerebro.adddata(data)

    # Set initial cash for simulation
    cerebro.broker.setcash(100000.0)

    # Run backtest
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Optionally, plot the results using Backtrader’s built-in plotting
    cerebro.plot(style='candlestick')
