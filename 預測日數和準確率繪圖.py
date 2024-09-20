import pandas as pd
import talib


DataFolder = 'Ori_Data/'
Currency_data = pd.read_excel(DataFolder + 'USDtoTWD_Currency_data.xlsx', 
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
Currency_data['WMA'] = talib.WMA(df_close, 30) # 計算 MA5
Currency_data['STDDEV'] = talib.STDDEV (df_close, timeperiod = 5, nbdev = 1)
Currency_data['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS (df_open, df_high, 
                                                        df_low, df_close)


# 讀入其他資料進行合併
Fed_Funds_Rate = pd.read_excel(DataFolder + 'Fed_Funds_Rate.xlsx')  
USA_CPI = pd.read_excel(DataFolder + 'USA_CPI_Data.xlsx')  
USA_Unemployment_Rate = pd.read_excel(DataFolder + 'USA_Unemployment_Rate.xlsx')  
TW_CPI = pd.read_excel(DataFolder + 'TW_CPI.xlsx')
USA_GDP = pd.read_excel(DataFolder + 'USA_GDP.xlsx')
TW_Rate = pd.read_excel(DataFolder + 'TW_Rate.xlsx')
DXY_NYB = pd.read_excel(DataFolder + 'Dx-y_data.xlsx')
GOLD_data = pd.read_excel(DataFolder + 'Gold_data.xlsx')

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
end_date = '2020-12-31'

# 排除特定期間內的數據
df_merge.set_index('DATE', inplace = True)
df_merge.drop(df_merge.loc[:end_date].index, inplace = True)
print(df_merge.head())

# 計算差距欄位 (欄位之間)
df_merge['CPI_Delta'] = df_merge['CPIAUCNS'] - df_merge['TW_CPI'] # 兩國 CPI 差距

# 移除無用欄位
df_merge = df_merge.drop(columns = ['Volume', 'BOP', 'CDL3BLACKCROWS', 
                                    'Gold_Adj_Close'])



def classify_return(x):
    return 1 if x > 0 else 0  # 標示漲跌，大於0標示為漲(1)，小於0標示為跌(0)

def function(num): # 預測日數
    # 處理 y 資料
    pre_day = num
    df_merge[f'Next_{pre_day}Day_Return'] = \
        df_merge['Close'].diff(pre_day).shift(-pre_day) # 計算價格變化
    # =============================================================================
    # diff 函數，計算列中相鄰元素之間的差異。計算當前值與前指定時間點的值（pre_day）的差
    # shift 函數﹑移動要指定哪個目標資料，負數表示向上移動，反之向下
    # =============================================================================

    df_merge['LABEL'] = \
        df_merge[f'Next_{pre_day}Day_Return'].apply(
            classify_return) # 創造新的一列 LABEL 來記錄漲跌
    
    
feature_names = ['Gold_Close', 'Gold_High', 'CPIAUCNS', 'Gold_Open', 'UNRATE', 
                 'MA_20', 'MA_10', 'USD_Index_Growth_Rate', 'TW_CPI_Rate', 
                 'WILLR', 'Open', 'K', 'RSI_14', 'Gold_Volume', 
                 'Gold_Growth_Rate', 'FEDFUNDS', 'Bollinger Bands lower', 
                 'Bollinger Bands Upper', 'USA_GDP_Rate']

def split_stock_data(stock_data, label_column, 
                     test_size = 0.3, random_state = 42):
    X = stock_data[feature_names].values
    y = stock_data[label_column].values
    
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size = test_size, 
                         random_state = random_state)

    return X_train, X_test, y_train, y_test, feature_names


test_list = []
    
for i in range(60):
    function(i + 1)
    
    # 移除過去新增的欄位
    if (i + 1) > 1:
        df_merge = df_merge.drop(columns = [f'Next_{i}Day_Return'])
    
    #print(df_merge.head())
    df_merge.to_excel("data.xlsx", index = True) # 將整理好的資料存成 excel
    print("已將結果寫入檔案 data.xlsx")
    
    ones_count = (df_merge['LABEL'] == 1).sum()
    zero_count = (df_merge['LABEL'] == 0).sum()
    print(f"上漲數為 {ones_count} ({ones_count / df_merge['LABEL'].count() * 100:.1f} %)")
    print(f"下跌數為 {zero_count} ({zero_count / df_merge['LABEL'].count() * 100:.1f} %)")
    print(f"總特徵數為 {len(df_merge.columns)}")
    
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    import time


    # 讀取數據
    df = pd.read_excel('data.xlsx') 
    label_column = 'LABEL'

    # 分割資料
    trainX, testX, trainY, testY, feature_names = \
        split_stock_data(df, label_column)
    Xgboost = XGBClassifier()
    start_time = time.time()
    Xgboost.fit(trainX, trainY)
    training_time = time.time() - start_time

    test_predic = Xgboost.predict(testX) # 取得預測的結果
    test_acc = Xgboost.score(testX, testY)

    print(f'Xgboost({i + 1}) 測試集準確率 %.3f' % test_acc)
    print(f"訓練時間: {training_time // 60:.2f} 分 {training_time % 60:.2f} 秒")
    # 0.821
    
    test_list.append(test_acc)
    
    
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Microsoft JhengHei' # 設置中文字體
plt.figure(figsize = (12, 6))
all_acc = pd.Series(test_list)
a = all_acc.iloc[1:].plot(kind = 'line', color = 'green')

# 設置刻度字型大小
# 'axis' 設置影響 x 軸和 y 軸；'which' 設置影響主要刻度；'labelsize' 設置刻度字型大小
plt.tick_params(axis = 'both', which = 'major', labelsize = 14)

# 或者直接在 plt.yticks 設定參數 fontsize。
ytick = [i / 20 for i in range(10, 21)] # 注意到 range 的全部參數僅能輸入整數，因此要輸出小數數列需換個形式
plt.yticks(ytick, [str(int(i * 100)) + ' %' for i in ytick]) # 實際刻度值、顯示刻度值
plt.ylim(0.5, 1.05)

# 在圖上標註最大值
plt.annotate(f'{all_acc.idxmax()}日 ({all_acc.max() * 100:.1f} %)',
             xy = (all_acc.idxmax(), all_acc.max()),  # 註釋點的位置
             xytext = (all_acc.idxmax(), all_acc.max() - 0.1),  # 註釋文字的位置
             arrowprops = dict(facecolor = 'red', shrink = 0.2, 
                               headlength = 14),  # 設置箭頭的屬性
             fontsize = 16,
             color = 'red',
             ha = 'center')
# =============================================================================
# arrowprops 參數
# facecolor：箭頭的填充顏色。這裡設為 'red'。
# edgecolor：箭頭邊緣的顏色。這裡設為 'black'。
# shrink：箭頭縮小比例。值為 0.1 時箭頭會比正常情況長，根據需要調整。
# width：箭頭線條的寬度。值為 1 時箭頭線條寬度為 1 個像素。
# headwidth：箭頭頭部的寬度。增加此值會使箭頭頭部更寬。
# headlength：箭頭頭部的長度。增加此值會使箭頭頭部更長。
# =============================================================================


# 在圖上標註最小值
plt.annotate(f'{all_acc.iloc[1:].idxmin()}日 ({all_acc.iloc[1:].min() * 100:.1f} %)',
             xy = (all_acc.iloc[1:].idxmin(), all_acc.iloc[1:].min()),  # 註釋點的位置
             xytext = (all_acc.iloc[1:].idxmin() + 2, 
                       all_acc.iloc[1:].min() - 0.05),  # 註釋文字的位置
             arrowprops = dict(facecolor = 'blue', shrink = 0.1, 
                               shrinkA = 5, headlength = 14),  # 設置箭頭的屬性
             fontsize = 16,
             color = 'blue',
             va = 'center')

plt.title(f'預測日數準確度折線圖 【訓練資料期間 {str(df_merge.index[0])[:10]} - {str(df_merge.index[len(df_merge) - 1])[:10]}】', 
          fontsize = 18)
plt.xlabel('預測日數', fontsize = 15)
plt.ylabel('測試集 準確率', fontsize = 15)
plt.grid(axis = 'y')
plt.legend(['測試集 準確率'], loc = 'upper left', fontsize = 15)



