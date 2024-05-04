import pandas as pd
import numpy as np

df_15min = pd.read_csv("data/EURJPY_15 Mins_Bid_2003.08.04_2024.04.13.csv")
df_5min = pd.read_csv("data/EURJPY_5 Mins_Bid_2003.08.04_2024.04.13.csv")

df_15min['Time (EET)'] = pd.to_datetime(df_15min['Time (EET)'])
df_5min['Time (EET)'] = pd.to_datetime(df_5min['Time (EET)'])

_15min_samples = []

total_len = df_15min.shape[0]
for idx in range(total_len-2):

    first_row = df_15min.iloc[idx]
    second_row = df_15min.iloc[idx+1]
    timestamp_15min_first = df_15min.iloc[idx]['Time (EET)']
    timestamp_15min_second = df_15min.iloc[idx+1]['Time (EET)']

    corresponding_5min_rows = df_5min[
        (df_5min['Time (EET)'] >= timestamp_15min_first) &
        (df_5min['Time (EET)'] <= timestamp_15min_second)
        ]
    if not corresponding_5min_rows.empty:

        #print(f"For 15min row between dates: {first_row['Time (EET)']} {second_row['Time (EET)']}, High: {second_row['High']}, Low: {second_row['Low']}")
        _15min_high = second_row["High"]
        _15min_low = second_row["Low"]
        _15min_close = second_row["Close"]
        #print(corresponding_5min_rows, end="\n")
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
