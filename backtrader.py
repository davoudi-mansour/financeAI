import backtrader as bt
import datetime
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Fetch the data using yfinance
data = yf.download('AAPL', '2020-01-01', '2021-01-01')
print(data)
# Create features for the model
data['SMA10'] = data['Close'].rolling(window=10).mean()
data['SMA30'] = data['Close'].rolling(window=30).mean()
data['Return'] = data['Close'].pct_change()
data['Signal'] = 0

# Generate signals: 1 for buy, -1 for sell
data['Signal'][data['SMA10'] > data['SMA30']] = 1
data['Signal'][data['SMA10'] < data['SMA30']] = -1

# Drop NA values
data.dropna(inplace=True)

# Prepare the dataset
X = data[['SMA10', 'SMA30', 'Return']]
y = data['Signal']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict signals on the test set
y_pred = model.predict(X_test)

# Print the accuracy of the model
print('Model Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# Add predictions to the data
data['Predicted_Signal'] = model.predict(X)

# Create a custom strategy using the predicted signals
class RandomForestStrategy(bt.SignalStrategy):
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.last_action = None

    def next(self):
        current_signal = data['Predicted_Signal'].iloc[len(self) - 1]

        if current_signal == 1 and self.last_action != 'buy':
            self.buy()
            self.last_action = 'buy'
        elif current_signal == -1 and self.last_action != 'sell':
            self.sell()
            self.last_action = 'sell'

# Convert the data into a format compatible with Backtrader
data_feed = bt.feeds.PandasData(dataname=data)

# Create a cerebro entity
cerebro = bt.Cerebro()

# Add the strategy
cerebro.addstrategy(RandomForestStrategy)

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
