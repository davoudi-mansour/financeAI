import pandas as pd
import numpy as np
import plotly.graph_objects as go


df = pd.read_csv("data/EURJPY_Hourly_Bid_2003.08.04_2024.03.31.csv")

PEAK, VALLEY = 1, -1

def _identify_initial_pivot(X, up_thresh, down_thresh):
    """Quickly identify the X[0] as a peak or valley."""
    x_0 = X[0]
    max_x = x_0
    max_t = 0
    min_x = x_0
    min_t = 0
    up_thresh += 1
    down_thresh += 1

    for t in range(1, len(X)):
        x_t = X[t]

        if x_t / min_x >= up_thresh:
            return VALLEY if min_t == 0 else PEAK

        if x_t / max_x <= down_thresh:
            return PEAK if max_t == 0 else VALLEY

        if x_t > max_x:
            max_x = x_t
            max_t = t

        if x_t < min_x:
            min_x = x_t
            min_t = t

    t_n = len(X) - 1
    return VALLEY if x_0 < X[t_n] else PEAK

def peak_valley_pivots_candlestick(close, high, low, up_thresh, down_thresh):

    if down_thresh > 0:
        raise ValueError('The down_thresh must be negative.')

    initial_pivot = _identify_initial_pivot(close, up_thresh, down_thresh)

    t_n = len(close)
    pivots = np.zeros(t_n, dtype='i1')
    pivots[0] = initial_pivot

    up_thresh += 1
    down_thresh += 1

    trend = -initial_pivot
    last_pivot_t = 0
    last_pivot_x = close[0]
    for t in range(1, len(close)):
        if trend == -1:
            x = low[t]
            r = x / last_pivot_x
            if r >= up_thresh:
                pivots[last_pivot_t] = trend
                trend = 1
                last_pivot_x = high[t]
                last_pivot_t = t
            elif x < last_pivot_x:
                last_pivot_x = x
                last_pivot_t = t
        else:
            x = high[t]
            r = x / last_pivot_x
            if r <= down_thresh:
                pivots[last_pivot_t] = trend
                trend = -1
                # last_pivot_x = x
                last_pivot_x = low[t]
                last_pivot_t = t
            elif x > last_pivot_x:
                last_pivot_x = x
                last_pivot_t = t

    if last_pivot_t == t_n - 1:
        pivots[last_pivot_t] = trend
    elif pivots[t_n - 1] == 0:
        pivots[t_n - 1] = trend

    return pivots

pivots = peak_valley_pivots_candlestick(df.Close, df.High, df.Low, .01, -.01)
df['Pivots'] = pivots
df['Pivot Price'] = np.nan
df.loc[df['Pivots'] == 1, 'Pivot Price'] = df.High
df.loc[df['Pivots'] == -1, 'Pivot Price'] = df.Low

series = df['Pivot Price'].dropna()
consecutive_subtract = series - series.shift(1)
print(consecutive_subtract)
sorted_candidates = np.sort(consecutive_subtract[1:])

print(f"alpha parameter is {round(abs(sorted_candidates[int(len(sorted_candidates)/2)]), 2)}")

df["Date"] = df['Date']

fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'])])

df_diff = df['Pivot Price'].dropna().diff().copy()

fig.add_trace(
    go.Scatter(mode="lines+markers",
               x=df['Date'],
               y=df["Pivot Price"]
               ))

fig.update_layout(
    autosize=False,
    width=1000,
    height=800, )

fig.add_trace(go.Scatter(x=df['Date'],
                         y=df['Pivot Price'].interpolate(),
                         mode='lines',
                         line=dict(color='black')))

def annot(value):
    if np.isnan(value):
        return ''
    else:
        return value

j = 0
for i, p in enumerate(df['Pivot Price']):
    if not np.isnan(p):
        fig.add_annotation(dict(font=dict(color='rgba(0,0,200,0.8)', size=12),
                                x=df['Date'].iloc[i],
                                y=p,
                                showarrow=False,
                                text=annot(round(abs(df_diff.iloc[j]), 3)),
                                textangle=0,
                                xanchor='right',
                                xref="x",
                                yref="y"))
        j = j + 1

fig.update_xaxes(type='category')
fig.show()