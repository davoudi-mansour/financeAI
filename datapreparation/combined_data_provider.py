import os
import pandas as pd
import numpy as np
from datetime import datetime
import openpyxl
from ta import trend, volatility, momentum, volume

PAIR = "EURUSD"
CURRENCY = ["EUR", "USD"]
TimeFrame = "Hourly"

data_dir = os.path.join('./data', PAIR)
fundamental_dir = './fundamentalData'


def calculate_volatility_ratio():
    market_data_path = os.path.join(data_dir, PAIR + '_' + TimeFrame + ".csv")
    market_df = pd.read_csv(market_data_path)
    market_df['date'] = [datetime.strptime(date, '%Y.%m.%d %H:%M:%S') for date in market_df['Time']]
    market_df = market_df.drop('Time', axis=1)
    market_df = market_df.set_index('date')
    for currency in CURRENCY:
        curr_dir = os.path.join(data_dir, currency)
        if not os.path.exists(curr_dir):
            os.makedirs(curr_dir)
        fundamental_data_path = os.path.join(fundamental_dir, currency + 'EventsDeviation.xlsx')
        fundamentals_df = pd.read_excel(fundamental_data_path, sheet_name=None)

        out_file = os.path.join(curr_dir, PAIR + "_" + currency + "_" + TimeFrame + "VolatilityRatio.xlsx")
        wb = openpyxl.Workbook()
        wb.save(out_file)

        for event, df in fundamentals_df.items():
            # print(event)
            df = df.set_index('Date')
            for w in df.index:
                idx = market_df.index.get_indexer([w], method='ffill')
                i = idx[0]
                if i != -1:
                    prev_close = market_df.loc[market_df.index[i - 1], 'Close']
                    curr_high = market_df.loc[market_df.index[i], 'High']
                    curr_low = market_df.loc[market_df.index[i], 'Low']
                    curr_close = market_df.loc[market_df.index[i], 'Close']
                    next_high = market_df.loc[market_df.index[i + 1], 'High']
                    next_low = market_df.loc[market_df.index[i + 1], 'Low']
                    next_close = market_df.loc[market_df.index[i + 1], 'Close']

                    true_range_curr = abs(
                        max((curr_high - curr_low), (curr_high - prev_close), (curr_low - prev_close), key=abs))
                    true_range_next = abs(
                        max((next_high - next_low), (next_high - curr_close), (next_low - curr_close), key=abs))
                    true_range_prev_next = abs(
                        max((next_high - next_low), (next_high - prev_close), (next_low - prev_close), key=abs))

                    dir_curr = 1
                    dir_next = 1
                    dir_prev_next = 1
                    if curr_close - prev_close < 0:
                        dir_curr = dir_curr * -1
                    if next_close - curr_close < 0:
                        dir_next = dir_next * -1
                    if next_close - prev_close < 0:
                        dir_prev_next = dir_prev_next * -1

                    df.loc[w, 'True_Rng_Curr'] = true_range_curr * dir_curr
                    df.loc[w, 'True_Rng_Next'] = true_range_next * dir_next
                    df.loc[w, 'True_Rng_PrevNext'] = true_range_prev_next * dir_prev_next

            df['True_Rng_Avg_Curr'] = trend.sma_indicator(df['True_Rng_Curr'].abs(), window=14)
            df['True_Rng_Avg_Next'] = trend.sma_indicator(df['True_Rng_Next'].abs(), window=14)
            df['True_Rng_Avg_PrevNext'] = trend.sma_indicator(df['True_Rng_PrevNext'].abs(), window=14)
            df['Volatility_Ratio_Curr'] = df['True_Rng_Curr'] / df['True_Rng_Avg_Curr']
            df['Volatility_Ratio_Next'] = df['True_Rng_Next'] / df['True_Rng_Avg_Next']
            df['Volatility_Ratio_PrevNext'] = df['True_Rng_PrevNext'] / df['True_Rng_Avg_PrevNext']

            with pd.ExcelWriter(out_file, engine='openpyxl', mode='a') as writer:
                df.to_excel(writer, sheet_name=event)

    return


def align_data_with_fundamental(path):
    # market_data_path = os.path.join(data_dir, PAIR + '_' + TimeFrame + ".csv")
    market_df = pd.read_csv(path)
    market_df['date'] = pd.to_datetime(market_df['date'])
    # market_df = market_df.drop('Time', axis=1)
    market_df = market_df.set_index('date')

    for currency in CURRENCY:
        curr_dir = os.path.join(data_dir, currency)
        if not os.path.exists(curr_dir):
            os.makedirs(curr_dir)
        fundamental_data_path = os.path.join(curr_dir, PAIR + "_" + currency + "_" + TimeFrame + "VolatilityRatio.xlsx")
        fundamentals_df = pd.read_excel(fundamental_data_path, sheet_name=None)

        for event, df in fundamentals_df.items():
            # print(event)
            feature = currency + '_' + event + '_Dev'
            market_df[feature] = None
            df = df.set_index('Date')

            for w in df.index:
                deviation = df.loc[w, 'Dev_Fx_Imp']
                if w in market_df.index:
                    market_df.loc[w, feature] = deviation
                else:
                    i = market_df.index.get_indexer([w], method='ffill')
                    if i != -1:
                        market_df.loc[market_df.index[i], feature] = deviation

            # if pd.isnull(market_df.loc[market_df.index[0], feature]):
            #     market_df.loc[market_df.index[0], feature] = 0

            market_df[feature].fillna(0, inplace=True)
            # market_df = market_df.ffill(axis=0)
    # out_file = os.path.join(data_dir, PAIR + '_' + TimeFrame + "Fundamental.csv")
    market_df.to_csv(path)
    return


def shift_one_row_time(row, prev_datetime):
    if pd.isnull(prev_datetime[0]):
        return row

    prev = pd.to_datetime(prev_datetime[0])
    if (row['date'] - prev) > pd.Timedelta(hours=1):
        row['date'] = prev + pd.Timedelta(hours=1)
    prev_datetime[0] = row['date']
    return row


def shift_time(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    prev_datetime = [(pd.to_datetime(df['date'][0]) - pd.Timedelta(hours=1))]
    shifted_df = pd.DataFrame()
    shifted_df = shifted_df.append(df.apply(shift_one_row_time, axis=1, prev_datetime=prev_datetime))
    shifted_df.to_csv(path, index=False)


def calculate_technical_indicators():
    data_path = os.path.join(data_dir, PAIR + '_' + TimeFrame + ".csv")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['Time'])  #[datetime.strptime(date, '%Y.%m.%d %H:%M:%S') for date in df['Time']]
    df = df.drop('Time', axis=1)
    df = df.set_index('date')
    # Moving Average
    # df['MA'] = talib.MA(df['Close'])
    #
    # # Exponential Moving Average (EMA)
    # df['EMA'] = talib.EMA(df['Close'])
    #
    # # Moving Average Convergence Divergence (MACD)
    # macd, macdsignal, macdhist = talib.MACD(df['Close'])  # fastperiod=12, slowperiod=26, signalperiod=9
    # df['MACD'] = macd
    # # df['MACD_Signal'] = macdsignal
    # # df['MACD_Histogram'] = macdhist
    #
    # # Rate of Change
    # df['ROC'] = talib.ROC(df['Close'])
    #
    # # Momentum
    # df['Momentum'] = talib.MOM(df['Close'])
    #
    # # Relative Strength Index (RSI)
    # df['RSI'] = talib.RSI(df['Close'])
    #
    # # Bollinger Bands (BB)
    # upper, middle, lower = talib.BBANDS(df['Close'])
    # df['BBM'] = middle
    # df['BBH'] = upper
    # df['BBL'] = lower
    #
    # # Commodity Channel Index
    # df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'])
    #
    # # On-Balance Volume (OBV)
    # df['OBV'] = talib.OBV(df['Close'], df['Volume'])
    #
    # # Williams %R
    # df['Williams'] = talib.WILLR(df['High'], df['Low'], df['Close'])
    #
    # # Accumulation/Distribution Index (ADI)
    # df['ADI'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
    #
    # # Average True Range (ATR)
    # df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'])
    #
    # # Stochastic
    # df['STCH%K'] = talib.STOCH(df['High'], df['Low'], df['Close'])[0]
    # df['STCH%D'] = talib.STOCH(df['High'], df['Low'], df['Close'])[1]
    df['SD'] = df['Close'].rolling(window=20).std(ddof=0)
    df['AO'] = momentum.awesome_oscillator(df['High'], df['Low'])
    df['BBH'] = volatility.bollinger_hband(df['Close'])
    df['BBL'] = volatility.bollinger_lband(df['Close'])
    df['FI'] = volume.force_index(df['Close'], df['Volume'])
    df['FI5'] = df['FI'].rolling(window=5).mean()
    df['C%'] = df['Close'].pct_change()
    df['V%'] = df['Volume'].pct_change()
    df['NVI'] = volume.negative_volume_index(df['Close'], df['Volume'])
    df['SEMV'] = volume.sma_ease_of_movement(df['High'], df['Low'], df['Volume'])
    df['TSI'] = momentum.tsi(df['Close'])
    df['MFI'] = volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
    # df['VI'] = trend.vortex_indicator_pos()
    df['KST'] = trend.kst(df['Close'])
    df['KST9'] = trend.kst_sig(df['Close'], nsig=9)
    df['DPO'] = trend.dpo(df['Close'])
    df['ADX7'] = trend.adx(df['High'], df['Low'], df['Close'], window=7)
    df['ADX14'] = trend.adx(df['High'], df['Low'], df['Close'], window=14)
    df['SMA5'] = trend.sma_indicator(df['Close'], window=5)
    df['SMA10'] = trend.sma_indicator(df['Close'], window=10)
    df['SMA20'] = trend.sma_indicator(df['Close'], window=20)
    df['EMA6'] = trend.ema_indicator(df['Close'], window=6)
    df['EMA10'] = trend.ema_indicator(df['Close'], window=10)
    df['EMA14'] = trend.ema_indicator(df['Close'], window=14)
    df['MACD'] = trend.macd(df['Close'], window_slow=26, window_fast=12)
    df['RSI10'] = momentum.rsi(df['Close'], window=10)
    df['RSI14'] = momentum.rsi(df['Close'], window=14)
    df['CCI'] = trend.cci(df['High'], df['Low'], df['Close'])
    df['H-L'] = df['High'].sub(df['Low'])
    df['H-Cp'] = df['High'].sub(df['Close'].shift(1))
    df['L-Cp'] = df['Low'].sub(df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-Cp', 'L-Cp']].max(axis=1)
    df['ATR'] = volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['ROC'] = momentum.roc(df['Close'])
    df['Williams%R'] = momentum.williams_r(df['High'], df['Low'], df['Close'])
    df['OBV'] = volume.on_balance_volume(df['Close'], df['Volume'])

    df.dropna(inplace=True)

    out_file = os.path.join(data_dir, PAIR + '_' + TimeFrame + "Indicators&Fundamental.csv")
    df.to_csv(out_file)

    return out_file

def main():
    # calculate_volatility_ratio()
    path = calculate_technical_indicators()
    align_data_with_fundamental(path)
    # shift_time(os.path.join(data_dir, PAIR + '_' + TimeFrame + "Fundamental.csv"))
    shift_time(os.path.join(data_dir, PAIR + '_' + TimeFrame + "Indicators&Fundamental.csv"))

    return


if __name__ == "__main__":
    main()
