import pandas as pd
import numpy as np

hourly_data = pd.read_csv("data/EURJPY_Hourly_Bid_2003.08.04_2024.03.31.csv")
_15min_data = pd.read_csv("data/EURJPY_15 Mins_Bid_2003.08.04_2024.03.31.csv")

# Convert "Date" column to datetime objects
hourly_data['Time (EET)'] = pd.to_datetime(hourly_data['Time (EET)'])
_15min_data['Time (EET)'] = pd.to_datetime(_15min_data['Time (EET)'])

# Create an empty list to store the corresponding hourly rows
corresponding_15min_rows = []
hourly_samples = []
for index, hourly_row in hourly_data.iterrows():
    # Filter the hourly dataframe for the corresponding date
    current_row = hourly_row['Time (EET)']
    corresponding_15_min_rows = _15min_data[
        (_15min_data['Time (EET)'].dt.date == current_row.date()) &
        (_15min_data['Time (EET)'].dt.hour == current_row.hour)
        ]    # Print the corresponding hourly rows for the current daily row
    if not corresponding_15_min_rows.empty:
        print(f"For daily row with date: {hourly_row['Time (EET)']}, High: {hourly_row['High']}, Low: {hourly_row['Low']}")
        hourly_high = hourly_row["High"]
        hourly_low = hourly_row["Low"]
        print(corresponding_15_min_rows, end="\n")
        for idx, _15min_row in corresponding_15_min_rows.iterrows():
            _15min_high = _15min_row["High"]
            _15min_low = _15min_row["Low"]
            if hourly_high == _15min_high:
                hourly_samples.append(hourly_high)
                hourly_samples.append(hourly_low)
                break
            elif hourly_low == _15min_low:
                hourly_samples.append(hourly_low)
                hourly_samples.append(hourly_high)
                break

np.save('data/hourly_samples.npy', hourly_samples)