# =============================================================================
# 2 版僅處理 2024 年初至今（前一日）的最新資料（檔名有 '_Latest.XXX'）
# =============================================================================

import pandas as pd
import talib


DataFolder = 'Ori_Data/'
Currency_data = pd.read_excel(DataFolder + 'USDtoTWD_Currency_Data_Latest.xlsx', 
                              index_col = 'Date')  # 讀取匯率資料
Currency_data.drop(columns = ['Adj Close'], inplace = True)
df_open = Currency_data['Open']
df_close = Currency_data['Close']
df_high = Currency_data['High']
df_low = Currency_data['Low']


# 處理 x 資料
Currency_data['MA_5'] = talib.SMA(df_close, 5) # 計算 MA5
Currency_data['MA_10'] = talib.SMA(df_close, 10) # 計算 MA10
Currency_data['MA_20'] = talib.SMA(df_close, 20) # 計算 MA20
Currency_data['RSI_14'] = talib.RSI(df_close, 14) # 計算 RSI
macd, macdsignal, macdhist = talib.MACD(df_close, fastperiod = 12, 
                                        slowperiod = 26, 
                                        signalperiod = 9) # 計算 MACD
Currency_data['MACD'] = macd # 將 MACD 計算結果存回資料中
Currency_data['K'],  Currency_data['D'] = \
    talib.STOCH(df_high, df_low, df_close, fastk_period = 14, 
                slowk_period = 14, slowd_period = 3) # 計算 KD

upperband, middleband, lowerband = talib.BBANDS(df_close, 
                                          timeperiod = 5, 
                                          nbdevup = 2, nbdevdn = 2, 
                                          matype = 0)
Currency_data['Bollinger Bands Upper'] = upperband
Currency_data['Bollinger Bands Middle'] = middleband
Currency_data['Bollinger Bands lower'] = lowerband
Currency_data['CCI'] = talib.CCI(df_high, df_low, df_close, timeperiod = 14)
Currency_data['MOM'] = talib.MOM(df_close, timeperiod = 10)
Currency_data['BOP'] = talib.BOP(df_open, df_high, df_low, df_close)
Currency_data['WILLR'] = talib.WILLR(df_high, df_low, df_close, 
                                     timeperiod = 14)
Currency_data['SAR'] = talib.SAR(df_high, df_low)
Currency_data['AVGPRICE'] = talib.AVGPRICE(df_open, df_high, df_low, df_close)
Currency_data['WCLPRICE'] = talib.WCLPRICE(df_high, df_low, df_close)
Currency_data['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(df_close, 14)
Currency_data['WMA'] = talib.WMA(df_close,30) # 計算 MA5
Currency_data['STDDEV'] = talib.STDDEV (df_close, timeperiod=5, nbdev=1)
Currency_data['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS (df_open, df_high, 
                                                        df_low, df_close)


# 讀入其他資料進行合併
Fed_Funds_Rate = pd.read_excel(DataFolder + 'Fed_Funds_Rate_Latest.xlsx')  
USA_CPI = pd.read_excel(DataFolder + 'USA_CPI_Data_Latest.xlsx')  
USA_Unemployment_Rate = pd.read_excel(DataFolder + 'USA_Unemployment_Rate_Latest.xlsx')  
TW_CPI = pd.read_excel(DataFolder + 'TW_CPI_Latest.xlsx')
USA_GDP = pd.read_excel(DataFolder + 'USA_GDP_Latest.xlsx')
TW_Rate = pd.read_excel(DataFolder + 'TW_Rate_Latest.xlsx')
DXY_NYB = pd.read_excel(DataFolder + 'Dx-y_Data_Latest.xlsx')
GOLD_data = pd.read_excel(DataFolder + 'Gold_Data_Latest.xlsx')

# =============================================================================
# merge_asof，用於合併兩個數據框，其中一個數據框的時間戳（或排序列）可能在另一個
# 數據框中找不到完全對應的記錄。這時，可以根據時間戳的前向或後向對齊進行合併。
#
# 參數說明
# left: 左側數據欄。
# 
# right: 右側數據欄。
# 
# on: 要合併的列名（必須在兩個數據框中都有），這通常是時間戳列。
# 
# by: 可選，按指定列進行額外分組（例如，根據產品 ID 分組合併）。
# 
# left_by 和 right_by: 可選，指定按這些列進行合併的分組。
# 
# direction: 合併方向，有三個選項：
#     'backward'（預設）：從右側數據框中選擇小於等於左側數據框的最接近值。
#     'forward'：從右側數據框中選擇大於等於左側數據框的最接近值。
#     'nearest'：從右側數據框中選擇最接近的值。
# 
# tolerance: 可選，指定容忍度（以時間間隔為單位），用來限制合併的時間差。
# =============================================================================
df_merge = pd.merge_asof(Fed_Funds_Rate.sort_values('DATE'), 
                         USA_CPI.sort_values('DATE'), on = 'DATE') # 合併資料
df_merge = pd.merge_asof(df_merge.sort_values('DATE'), 
                         USA_Unemployment_Rate.sort_values('DATE'), 
                         on = 'DATE') # 合併資料

TW_CPI = TW_CPI.rename(columns = {'CPI': 'TW_CPI'}) # 欄位名稱調整
df_merge = pd.merge_asof(df_merge.sort_values('DATE'), 
                         TW_CPI.sort_values('DATE'), on = 'DATE') # 合併資料

USA_GDP = USA_GDP.rename(columns = {'GDP': 'US_GDP'}) # 欄位名稱調整
df_merge = pd.merge_asof(df_merge.sort_values('DATE'), 
                         USA_GDP.sort_values('DATE'), on = 'DATE') # 合併資料

df_merge = pd.merge_asof(df_merge.sort_values('DATE'), 
                         TW_Rate.sort_values('DATE'), on = 'DATE') # 合併資料


DXY_NYB = DXY_NYB.rename(columns = {'Date': 'DATE', 'Close': 'USD_Index', 
                                    'Growth Rate': 'USD_Index_Growth_Rate'}) # 美元指數小寫改大寫
df_merge = pd.merge_asof(df_merge.sort_values('DATE'), 
                         DXY_NYB.sort_values('DATE'), on = 'DATE') # 合併資料

GOLD_data = GOLD_data.rename(columns = {'Date': 'DATE', 'Open': 'Gold_Open', 
                                        'High': 'Gold_High', 
                                        'Low': 'Gold_Low', 
                                        'Close': 'Gold_Close',
                                        'Adj Close': 'Gold_Adj_Close',
                                        'Volume': 'Gold_Volume',
                                        'Growth Rate': 'Gold_Growth_Rate'}) # 黃金改大寫
df_merge = pd.merge_asof(df_merge.sort_values('DATE'), 
                         GOLD_data.sort_values('DATE'), on = 'DATE') # 合併資料

Currency_data = Currency_data.reset_index()
Currency_data = Currency_data.rename(columns = {'Date': 'DATE'})
df_merge = pd.merge_asof(Currency_data.sort_values('DATE'), 
                         df_merge.sort_values('DATE'), on = 'DATE') # 合併資料

# 計算兩筆資料間差距 (前後或者上下之間)
df_merge['FEDFUNDS_Delta'] = df_merge['FEDFUNDS'].pct_change(periods = 21)


# 因資料特定欄位計算有回朔需求而向前推進抓取時間，設定要排除的期間
end_date = '2024-01-01'

# 排除特定期間內的數據
df_merge.set_index('DATE', inplace = True)
df_merge.drop(df_merge.loc[:end_date].index, inplace = True)
print(df_merge.head())

# 計算差距欄位 (欄位之間)
df_merge['CPI_Delta'] = df_merge['CPIAUCNS'] - df_merge['TW_CPI'] # 兩國 CPI 差距

# 移除無用欄位
df_merge = df_merge.drop(columns = ['Volume', 'BOP', 'CDL3BLACKCROWS', 
                                    'Gold_Adj_Close'])


# 處理 y 資料
pre_day = 5
df_merge[f'Next_{pre_day}Day_Return'] = \
    df_merge['Close'].diff(pre_day).shift(-pre_day) # 計算價格變化
# =============================================================================
# diff 函數，計算列中相鄰元素之間的差異。計算當前值與前指定時間點的值（pre_day）的差
# shift 函數﹑移動要指定哪個目標資料，負數表示向上移動，反之向下
# =============================================================================

def classify_return(x):
    return 1 if x > 0 else 0  # 標示漲跌，大於 0 標示為漲(1)，小於 0 標示為跌(0)

df_merge['LABEL'] = \
    df_merge[f'Next_{pre_day}Day_Return'].apply(
        classify_return) # 創造新的一列 LABEL 來記錄漲跌

print(df_merge.head())
print(df_merge.tail())
df_merge.to_excel("data_Latest.xlsx") # 將整理好的資料存成 excel
print("已將結果寫入檔案 data_Latest.xlsx")

ones_count = (df_merge['LABEL'] == 1).sum()
zero_count = (df_merge['LABEL'] == 0).sum()
print(f"上漲數為 {ones_count} ({ones_count / df_merge['LABEL'].count() * 100:.1f} %)")
print(f"下跌數為 {zero_count} ({zero_count / df_merge['LABEL'].count() * 100:.1f} %)")
print(f"總特徵數為 {len(df_merge.columns)}")

