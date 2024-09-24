import pandas as pd
from scipy import stats
import openpyxl
import os

PAIR = 'EURUSD'
CURRENCY = 'EUR'
TimeFrame = '15Mins'

data_dir = os.path.join('./data', PAIR)


def correlation_for_economic_data():
    df = pd.read_excel(os.path.join(data_dir, CURRENCY, PAIR + "_" + CURRENCY + "_" + TimeFrame + "VolatilityRatio.xlsx"), sheet_name=None)
    out_file = os.path.join(data_dir, PAIR + "_" + CURRENCY + "_" + TimeFrame + "Correlation.xlsx")
    wb = openpyxl.Workbook()
    wb.save(out_file)
    corr_between_list = [{'deviation_col':'Dev_Fx_Imp', 'volatility_col':'Volatility_Ratio_Curr', 'corr_between':'Dev_Fx_Imp_Vol_Curr'},
                         {'deviation_col':'Dev_Fx_Imp', 'volatility_col':'Volatility_Ratio_Next', 'corr_between':'Dev_Fx_Imp_Vol_Next'},
                         {'deviation_col':'Dev_Fx_Imp', 'volatility_col':'Volatility_Ratio_PrevNext', 'corr_between':'Dev_Fx_Imp_Vol_PrevNext'},
                         {'deviation_col':'Dev_Fx', 'volatility_col':'Volatility_Ratio_Curr', 'corr_between':'Dev_Fx_Vol_Curr'},
                         {'deviation_col':'Dev_Fx', 'volatility_col':'Volatility_Ratio_Next', 'corr_between':'Dev_Fx_Vol_Next'},
                         {'deviation_col':'Dev_Fx', 'volatility_col':'Volatility_Ratio_PrevNext', 'corr_between':'Dev_Fx_Vol_PrevNext'},
                         {'deviation_col':'Dev_MinMax_Imp', 'volatility_col':'Volatility_Ratio_Curr', 'corr_between':'Dev_MinMax_Imp_Vol_Curr'},
                         {'deviation_col':'Dev_MinMax_Imp', 'volatility_col':'Volatility_Ratio_Next', 'corr_between':'Dev_MinMax_Imp_Vol_Next'},
                         {'deviation_col':'Dev_MinMax_Imp', 'volatility_col':'Volatility_Ratio_PrevNext', 'corr_between':'Dev_MinMax_Imp_Vol_PrevNext'},
                         {'deviation_col':'Dev_MinMax', 'volatility_col':'Volatility_Ratio_Curr', 'corr_between':'Dev_MinMax_Vol_Curr'},
                         {'deviation_col': 'Dev_MinMax', 'volatility_col': 'Volatility_Ratio_Next', 'corr_between': 'Dev_MinMax_Vol_Next'},
                         {'deviation_col': 'Dev_MinMax', 'volatility_col': 'Volatility_Ratio_PrevNext', 'corr_between': 'Dev_MinMax_Vol_PrevNext'}]

    cols = ['Event', 'Correlation', 'P_Value']
    for item in corr_between_list:
        corr_dict = []
        for event, sheet in df.items():
            print('EVENT : ', event)
            # sheet.replace([np.inf, -np.inf], None, inplace=True)
            sheet.dropna(inplace=True)
            deviation = sheet[item['deviation_col']]
            volatility_ratio = sheet[item['volatility_col']]
            pearson_res = stats.pearsonr(deviation, volatility_ratio)
            correlation = round(pearson_res[0], 4)
            p_value = round(pearson_res[1], 4)

            print(item['corr_between']+' :    Corr = {}, P = {}'.format(correlation, p_value))

            vals = [event, correlation,  p_value]
            corr_dict.append(dict(zip(cols, vals)))

            print('+---------------------------------------------+\n')

        corr_df = pd.DataFrame(corr_dict)

        with pd.ExcelWriter(out_file, engine='openpyxl', mode='a') as writer:
            corr_df.to_excel(writer, sheet_name=item['corr_between'], index=False)


def correlation_for_technical_data():
    df = pd.read_csv(os.path.join(data_dir, PAIR + "_" + TimeFrame + "Indicators&Fundamental.csv"))
    out_file = os.path.join(data_dir, PAIR + "_" + TimeFrame + "TechnicalCorrelation.csv")

    df['Target'] = df['High'].shift(-1)
    technical_df = df[['SD', 'AO', 'BBH', 'BBL', 'FI', 'FI5', 'C%', 'V%', 'NVI', 'SEMV', 'TSI', 'MFI', 'KST', 'KST9',
                       'DPO', 'ADX7', 'ADX14', 'SMA5', 'SMA10', 'SMA20', 'EMA6', 'EMA10', 'EMA14', 'MACD', 'RSI10',
                       'RSI14', 'CCI', 'H-L', 'H-Cp', 'L-Cp', 'TR', 'ATR', 'ROC', 'Williams%R', 'OBV']]
    target = df['Target']
    target.iloc[-1] = target.iloc[-2]
    cols = ['Indicator', 'Correlation', 'P_Value']
    corr_dict = []
    for item in technical_df:
        pearson_res = stats.pearsonr(df[item], target)
        correlation = round(pearson_res[0], 4)
        p_value = round(pearson_res[1], 4)

        print(item + ' :    Corr = {}, P = {}'.format(correlation, p_value))

        vals = [item, correlation, p_value]
        corr_dict.append(dict(zip(cols, vals)))

        print('+---------------------------------------------+\n')

    corr_df = pd.DataFrame(corr_dict)
    corr_df.to_csv(out_file, index=False)


if __name__ == "__main__":
    # correlation_for_economic_data()
    correlation_for_technical_data()
