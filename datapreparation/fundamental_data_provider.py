import os
import pandas as pd
import numpy as np
from datetime import datetime
import openpyxl

CURRENCY = 'JPY'

fundamental_data_path = './fundamentalData'


def separate_each_currency_events():
    path = os.path.join(fundamental_data_path, 'events.csv')
    df = pd.read_csv(path)
    currency_df = df.loc[df['Currency'] == CURRENCY]
    out_file = os.path.join(fundamental_data_path, CURRENCY + 'Events.csv')
    currency_df.to_csv(out_file, index=False)
    return


def separate_speech_events():
    path = os.path.join(fundamental_data_path, CURRENCY + 'Events.csv')
    df = pd.read_csv(path)
    speech_df = df[df['Actual'].isna()]
    df = df[df['Actual'].notna()]
    df.to_csv(path, index=False)
    speech_file = os.path.join(fundamental_data_path, CURRENCY + 'SpeechEvents.csv')
    speech_df.to_csv(speech_file, index=False)
    return


def merge_date_and_time():
    path = os.path.join(fundamental_data_path, CURRENCY + 'Events.csv')
    df = pd.read_csv(path)
    df['DateTime'] = df['Date'] + ' ' + df['Time']
    df = df.drop('Date', axis=1)
    df = df.drop('Time', axis=1)
    df['Date'] = [datetime.strptime(date, '%Y-%m-%d %H:%M') for date in df['DateTime']]
    df = df.drop('DateTime', axis=1)
    df.to_csv(path, index=False)
    return


def categorize_related_events():
    event = "Monetary Base (YoY)"
    event1 = "Housing Starts (YoY)"
    event2 = "National CPI ex Fresh Food (YoY)"
    event3 = "Employment Change"
    event4 = "Claimant Count Change"
    event5 = "Claimant Count Rate"
    event6 = "Retail Price Index (YoY)"
    event7 = "Producer Price Index - Output (YoY) n.s.a"
    event8 = "Producer Price Index - Output (MoM) n.s.a"
    event9 = "Producer Price Index - Input (YoY) n.s.a"
    event10 = "Producer Price Index - Input (MoM) n.s.a"

    path = os.path.join(fundamental_data_path, CURRENCY + 'Events.csv')
    df = pd.read_csv(path)
    # event_df = df[df['Event'] == event]
    # event_df = df[(df['Event'] == event) | (df['Event'] == event1)]
    # event_df = df[(df['Event'] == event) | (df['Event'] == event1) | (df['Event'] == event2)]
    # event_df = df[(df['Event'] == event) | (df['Event'] == event1) | (df['Event'] == event2) | (df['Event'] == event3)]
    # event_df = df[(df['Event'] == event) | (df['Event'] == event1) | (df['Event'] == event2) | (df['Event'] == event3) | (df['Event'] == event4)]
    # event_df = df[(df['Event'] == event) | (df['Event'] == event1) | (df['Event'] == event2) | (df['Event'] == event3) | (df['Event'] == event4) | (df['Event'] == event5)]
    # event_df = df[(df['Event'] == event) | (df['Event'] == event1) | (df['Event'] == event2) | (df['Event'] == event3) | (df['Event'] == event4) | (df['Event'] == event5) | (df['Event'] == event6) | (df['Event'] == event7) | (df['Event'] == event8)]
    event_df = df[(df['Event'] == event) | (df['Event'] == event1) | (df['Event'] == event2) | (df['Event'] == event3) | (df['Event'] == event4) | (df['Event'] == event5) | (df['Event'] == event6) | (df['Event'] == event7) | (df['Event'] == event8) | (df['Event'] == event9) | (df['Event'] == event10)]
    out_file = os.path.join(fundamental_data_path, 'categorized' + CURRENCY + 'Events.xlsx')
    with pd.ExcelWriter(out_file, engine='openpyxl', mode='a') as writer:
        event_df.to_excel(writer, sheet_name=event.replace(" ", ""), index=False)

    return


def separate_all_events():
    # country = "Germany"
    # country = "Eurozone"
    event = "Monetary Base (YoY)"
    sheet = "MBY"
    path = os.path.join(fundamental_data_path, CURRENCY + 'Events.csv')
    df = pd.read_csv(path)
    event_df = df[(df['Event'] == event)]  # & (df['Country'] == country)
    out_file = os.path.join(fundamental_data_path, 'separated' + CURRENCY + 'Events.xlsx')
    with pd.ExcelWriter(out_file, engine='openpyxl', mode='a') as writer:
        event_df.to_excel(writer, sheet_name=sheet, index=False)

    return


def calculate_deviation():
    path = os.path.join(fundamental_data_path, 'separated' + CURRENCY + 'Events.xlsx')
    df = pd.read_excel(path, sheet_name=None)

    out_file = os.path.join(fundamental_data_path, CURRENCY + 'EventsDeviation.xlsx')
    wb = openpyxl.Workbook()
    wb.save(out_file)

    for event, df in df.items():
        df['Diff_ActCons'] = None
        df['Std'] = None
        df['Dev_Fx'] = None
        df['Dev_Fx_Imp'] = None
        for i, row in df.iterrows():
            i = int(i)
            actual = row['Actual']
            consensus = row['Consensus']
            previous = row['Previous']

            if not pd.isna(actual):
                actual = float(''.join(x for x in str(actual) if x.isdigit() or x in '-.'))
            if not pd.isna(consensus):
                consensus = float(''.join(x for x in str(consensus) if x.isdigit() or x in '-.'))
            if not pd.isna(previous):
                previous = float(''.join(x for x in str(previous) if x.isdigit() or x in '-.'))

            diff = actual - consensus
            df.iloc[i, df.columns.get_loc('Diff_ActCons')] = diff
            imp = df.iloc[i, df.columns.get_loc('Volatility')]
            if i >= 5:
                prev_diff = df.iloc[i - 5:i, df.columns.get_loc('Diff_ActCons')]
                diff = df.iloc[i, df.columns.get_loc('Diff_ActCons')]
                std = np.std(prev_diff)
                dev_fx = 0.0
                if std != 0:
                    dev_fx = round(diff / std, 4)
                dev_fx_imp = dev_fx * imp
                df.iloc[i, df.columns.get_loc('Std')] = std
                df.iloc[i, df.columns.get_loc('Dev_Fx')] = dev_fx
                df.iloc[i, df.columns.get_loc('Dev_Fx_Imp')] = dev_fx_imp

        with pd.ExcelWriter(out_file, engine='openpyxl', mode='a') as writer:
            df.to_excel(writer, sheet_name=event, index=False)

    return


def main():
    # separate_each_currency_events()
    # separate_speech_events()
    # merge_date_and_time()
    # categorize_related_events()
    # separate_all_events()
    calculate_deviation()
    return


if __name__ == "__main__":
    main()
