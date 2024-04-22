import pandas as pd
from scipy import stats
import openpyxl
import os

PAIR = 'EURUSD'
CURRENCY = 'EUR'
TimeFrame = '5Mins'

data_dir = os.path.join('./data', PAIR, CURRENCY)

df = pd.read_excel(os.path.join(data_dir, PAIR + "_" + CURRENCY + "_" + TimeFrame + "VolatilityRatio.xlsx"), sheet_name=None)

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

