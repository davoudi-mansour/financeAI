import pandas as pd
import plotly.graph_objects as go

# Sample data
df = pd.read_csv('result/output_result.csv')

# Convert 'Time (EET)' to datetime
df['Time (EET)'] = pd.to_datetime(df['Time (EET)'], format='%Y.%m.%d %H:%M:%S')

# Create the candlestick chart
fig = go.Figure(data=[go.Candlestick(x=df['Time (EET)'],
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'],
                                     name='Candlesticks')])

# Add green markers for buy signals (signal == 1)
buy_signals = df[df['signal'] == 1]
fig.add_trace(go.Scatter(x=buy_signals['Time (EET)'],
                         y=buy_signals['Close'],
                         mode='markers',
                         marker=dict(color='blue', size=4),
                         name='Buy Signal'))

# Add red markers for sell signals (signal == -1)
sell_signals = df[df['signal'] == -1]
fig.add_trace(go.Scatter(x=sell_signals['Time (EET)'],
                         y=sell_signals['Close'],
                         mode='markers',
                         marker=dict(color='yellow', size=4),
                         name='Sell Signal'))

# Update layout
fig.update_layout(title='Bitcoin Candlestick Chart with Buy/Sell Signals',
                  xaxis_title='Time (EET)',
                  yaxis_title='Price (USD)',
                  xaxis_rangeslider_visible=False)

# Show the plot
fig.show()
