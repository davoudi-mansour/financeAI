import pandas as pd
import numpy as np

daily_data = pd.read_csv("data/EURJPY_Daily.csv")
hourly_data = pd.read_csv("data/EURJPY_4 Hours.csv")

# Convert "Date" column to datetime objects
daily_data['Date'] = pd.to_datetime(daily_data['Date'])
hourly_data['Date'] = pd.to_datetime(hourly_data['Date'])

# Create an empty list to store the corresponding hourly rows
corresponding_hourly_rows = []
all_samples = []
for index, daily_row in daily_data.iterrows():
    # Filter the hourly dataframe for the corresponding date
    corresponding_hourly_rows = hourly_data[hourly_data['Date'].dt.date == daily_row['Date'].date()]
    # Print the corresponding hourly rows for the current daily row
    if not corresponding_hourly_rows.empty:
        print(f"For daily row with date {daily_row['Date']}:")
        daily_high = daily_row["High"] = daily_row["High"]
        daily_low = daily_row["Low"]
        print(corresponding_hourly_rows, end="\n")
        for idx, hourly_row in corresponding_hourly_rows.iterrows():
            hourly_high = hourly_row["High"]
            hourly_low = hourly_row["Low"]
            if daily_high == hourly_high:
                all_samples.append(daily_high)
                all_samples.append(daily_low)
                break
            elif daily_low == hourly_low:
                all_samples.append(daily_low)
                all_samples.append(daily_high)
                break


np.save('data/hourly_samples.npy', all_samples)