import pandas as pd
import numpy as np

df_15min = pd.read_csv("data/EURJPY_15 Mins_Bid_2003.08.04_2024.04.13.csv")
df_5min = pd.read_csv("data/EURJPY_5 Mins_Bid_2003.08.04_2024.04.13.csv")

df_15min['Time (EET)'] = pd.to_datetime(df_15min['Time (EET)'])
df_5min['Time (EET)'] = pd.to_datetime(df_5min['Time (EET)'])

df_15min.set_index('Time (EET)', inplace=True)
df_5min.set_index('Time (EET)', inplace=True)

_15min_samples = []

for idx, _15min_row in df_15min.iterrows():
    corresponding_5min_rows = df_5min.loc[idx:idx + pd.Timedelta(minutes=15)]
    print(f"For 15min row with date: {idx}, High: {_15min_row['High']}, Low: {_15min_row['Low']}")
    _15min_high = _15min_row["High"]
    _15min_low = _15min_row["Low"]
    _15min_close = _15min_row["Close"]

    print(corresponding_5min_rows, end="\n")
    for idx, _5min_row in corresponding_5min_rows.iterrows():
        _5min_high = _5min_row["High"]
        _5min_low = _5min_row["Low"]
        if _15min_high == _5min_high:
            _15min_samples.append(_15min_high)
            _15min_samples.append(_15min_low)
            _15min_samples.append(_15min_close)
            break
        elif _15min_low == _5min_low:
            _15min_samples.append(_15min_low)
            _15min_samples.append(_15min_high)
            _15min_samples.append(_15min_close)
            break

np.save('data/_15min_samples.npy', _15min_samples)
