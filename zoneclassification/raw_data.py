import pandas as pd
import numpy as np

df_major = pd.read_csv("data/BTCUSD_Hourly_Bid_2023.01.01_2024.01.01.csv")
df_minor = pd.read_csv("data/BTCUSD_15 Mins_Bid_2023.01.01_2024.01.01.csv")

df_major['Time (EET)'] = pd.to_datetime(df_major['Time (EET)'])
df_minor['Time (EET)'] = pd.to_datetime(df_minor['Time (EET)'])

major_samples = []

total_len = df_major.shape[0]
for idx in range(total_len-2):

    first_row = df_major.iloc[idx]
    second_row = df_major.iloc[idx+1]
    timestamp_15min_first = df_major.iloc[idx]['Time (EET)']
    timestamp_15min_second = df_major.iloc[idx+1]['Time (EET)']

    corresponding_minor_rows = df_minor[
        (df_minor['Time (EET)'] >= timestamp_15min_first) &
        (df_minor['Time (EET)'] <= timestamp_15min_second)
        ]
    if not corresponding_minor_rows.empty:
        major_high = second_row["High"]
        major_low = second_row["Low"]
        major_close = second_row["Close"]
        for idx, _minor_row in corresponding_minor_rows.iterrows():
            minor_high = _minor_row["High"]
            minor_low = _minor_row["Low"]
            if major_high == minor_high:
                major_samples.append(major_high)
                major_samples.append(major_low)
                major_samples.append(major_close)
                break
            elif major_low == minor_low:
                major_samples.append(major_low)
                major_samples.append(major_high)
                major_samples.append(major_close)
                break

np.save('data/major_samples.npy', major_samples)