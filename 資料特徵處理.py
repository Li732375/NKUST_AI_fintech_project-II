import pandas as pd
import talib


Currency_data = pd.read_excel('TWD%3DX_Currency_data.xlsx', 
                              index_col = 'Date')  # 讀取匯率資料

missing_values = Currency_data.isnull().sum() # 檢查每一列是否有空值

#print(missing_values)

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
                                          timeperiod=5, 
                                          nbdevup=2, nbdevdn=2, 
                                          matype=0)
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
Currency_data['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(df_close,14)
Currency_data['WMA'] = talib.WMA(df_close,30) # 計算 MA5
Currency_data['STDDEV'] = talib.STDDEV (df_close, timeperiod=5, nbdev=1)
Currency_data['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS (df_open, df_high, df_low, df_close)


columns_to_shift = ['Close', 'MA_5', 'MA_10', 'MA_20', 'RSI_14', 'MACD', 
                    'K', 'D','Bollinger Bands Upper', 
                    'Bollinger Bands Middle', 'Bollinger Bands lower',
                    'CCI', 'MOM', 'BOP','WILLR','SAR','AVGPRICE','LINEARREG_ANGLE',
                    'WMA','STDDEV','CDL3BLACKCROWS'] # 選取需要進行處理的欄位名稱


# =============================================================================
# # 參考前 5(週), 10(雙週), 15(三週), 20(月) 個交易日作為特徵相關參考
# for period in range(5, 21,5): # 運用迴圈帶入前 N 期收盤價
#         for column in columns_to_shift: # 運用迴圈走訪所選的欄位名稱
#             Currency_data[f'{column}_{period}'] = \
#                 Currency_data[column].shift(period) # 運用.shift()方法取得收盤價
# =============================================================================


# 因資料特定欄位計算有回朔需求而向前推進抓取時間，設定要排除的期間
start_date = '2019-01-01'
end_date   = '2019-12-31'

# 排除特定期間內的數據
Currency_data.drop(Currency_data.
                   loc[start_date : end_date].index, inplace = True)
print(Currency_data.head())

# 讀入其他資料進行合併
Fed_Funds_Rate = pd.read_excel('Fed_Funds_Rate.xlsx')  
USA_CPI = pd.read_excel('USA_CPI_Data.xlsx')  
USA_Unemployment_Rate = pd.read_excel('USA_Unemployment_Rate.xlsx')  
TW_CPI = pd.read_excel('TW_CPI.xlsx')
USA_GDP = pd.read_excel('USA_GDP.xlsx')
TW_Rate = pd.read_excel('TW_Rate.xlsx')
DXY_NYB = pd.read_excel('dxy_data.xlsx')
GOLD_data = pd.read_excel('gold_data.xlsx')

# =============================================================================
# merge_asof，用於合併兩個數據框，其中一個數據框的時間戳（或排序列）可能在另一個數據框中找不到完全對應的記錄。這時，可以根據時間戳的前向或後向對齊進行合併。
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
df_merge = pd.merge_asof(df_merge.sort_values('DATE'), 
                         TW_CPI.sort_values('DATE'), on = 'DATE') # 合併資料
df_merge = pd.merge_asof(df_merge.sort_values('DATE'), 
                         USA_GDP.sort_values('DATE'), on = 'DATE') # 合併資料
df_merge = pd.merge_asof(df_merge.sort_values('DATE'), 
                         TW_Rate.sort_values('DATE'), on = 'DATE') # 合併資料

DXY_NYB = DXY_NYB.rename(columns = {'Date': 'DATE'}) # 美元指數小寫改大寫
df_merge = pd.merge_asof(df_merge.sort_values('DATE'), 
                         DXY_NYB.sort_values('DATE'), on = 'DATE') # 合併資料

GOLD_data = GOLD_data.rename(columns = {'Date': 'DATE'}) # 黃金改大寫
df_merge = pd.merge_asof(df_merge.sort_values('DATE'), 
                         GOLD_data.sort_values('DATE'), on = 'DATE') # 合併資料

Currency_data = Currency_data.reset_index()
Currency_data = Currency_data.rename(columns = {'Date': 'DATE'})
df_merge = pd.merge_asof(Currency_data.sort_values('DATE'), 
                         df_merge.sort_values('DATE'), on = 'DATE') # 合併資料

# 計算差距欄位
df_merge['CPI Delta'] = df_merge['CPIAUCNS'] - df_merge['CPI']


# 處理 y 資料
pre_day = 5
df_merge[f'Next_{pre_day}Day_Return'] = \
    df_merge['Close'].diff(pre_day).shift(-pre_day) # 計算價格變化
# =============================================================================
# diff 函數，計算列中相鄰元素之間的差異。計算當前值與前指定時間點的值（pre_day）的差
# shift 函數﹑移動要指定哪個目標資料，負數表示向上移動，反之向下
# =============================================================================

def classify_return(x):
    return 1 if x > 0 else 0  # 標示漲跌，大於0標示為漲(1)，小於0標示為跌(0)

df_merge['LABEL'] = \
    df_merge[f'Next_{pre_day}Day_Return'].apply(
        classify_return) # 創造新的一列 LABEL 來記錄漲跌

print(df_merge.head())
df_merge.to_excel("data.xlsx", index = False) # 將整理好的資料存成 excel
print("已將結果寫入檔案 data.xlsx")

ones_count = (df_merge['LABEL'] == 1).sum()
zero_count = (df_merge['LABEL'] == 0).sum()
print(f"上漲數為 {ones_count}")
print(f"下跌數為 {zero_count}")
print(f"總特徵數為 {len(df_merge.columns)}")

