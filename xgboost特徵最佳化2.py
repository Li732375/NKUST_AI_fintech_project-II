# =============================================================================
# 相較於前版，2 嘗試臨時新增欄位 Index 供資料回朔，但熱力圖的部分仍然不變
# =============================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
import time


# 讀取數據
df = pd.read_excel('data.xlsx')
df['Index'] = range(len(df))

#print(df.columns)
feature_names = ['Gold_Close', 'Gold_High', 'CPIAUCNS', 'Gold_Open', 'UNRATE', 'MA_20', 
                 'MA_10', 'USD_Index_Growth_Rate', 'TW_CPI_Rate', 
                 'WILLR', 'Open', 'K', 'RSI_14', 'Gold_Volume', 
                 'Gold_Growth_Rate', 'FEDFUNDS', 'Bollinger Bands lower', 
                 'Bollinger Bands Upper', 'USA_GDP_Rate', 'Index']
# 0.821
    
def split_stock_data(stock_data, label_column, delete_column, test_size = 0.3, 
                     random_state = 42):
    X = stock_data[feature_names].values
    y = stock_data[label_column].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = test_size, 
                                                        random_state = 
                                                        random_state)

    return X_train, X_test, y_train, y_test, feature_names

label_column = 'LABEL'
delete_column = ['LABEL', 'Next_5Day_Return']
accuracies = []

# 分割資料
trainX, testX, trainY, testY, feature_names = split_stock_data(df, label_column, 
                                                delete_column)
Xgboost = XGBClassifier()
start_time = time.time()
Xgboost.fit(trainX, trainY)
training_time = time.time() - start_time

test_predic = Xgboost.predict(testX) # 取得預測的結果
test_acc = Xgboost.score(testX, testY)


# 進行 XOR 運算
xor_result = np.bitwise_xor(test_predic, testY)
# =============================================================================
# test_predic 和 testY 之間的 XOR 運算可以幫助你找出模型預測錯誤的情況：
# 
# 如果 test_predic[i] 和 testY[i] 相同（即模型預測正確），則 XOR 結果為 0。
# 如果 test_predic[i] 和 testY[i] 不同（即模型預測錯誤），則 XOR 結果為 1。
# =============================================================================
print(f"結果資料型態 {type(xor_result)}")
print(f"結果資料筆數 {len(xor_result)}")
#print(f"XOR 運算結果:\n{xor_result}")
    
print('Xgboost測試集準確率 %.3f' % test_acc)
print(f"訓練時間: {training_time // 60:.0f} 分 {training_time % 60:.2f} 秒")
# 0.821


# 這裡本圖僅參考分布，不具實際應用意義，因 train_test_split 抽選資料為隨機抽取(並
# 不連續)，並且輸出的資料完全不可回朔，僅能自行添加欄位回朔，不利於本繪圖的時間資料。
import matplotlib.colors as mcolors
import matplotlib.patches as patches


def heatmap_darw(result):
    num_columns = 5 # 每列 5 筆數據
    num_rows = (len(result) + num_columns - 1) // num_columns # 計算需要的行數
    data_padded = np.pad(result, 
                         (0, num_rows * num_columns - len(result)), 
                         mode = 'constant', constant_values = -1) # 補齊數據
    
    # 重塑為每列 5 筆數據的二維矩陣
    result_2d = data_padded.reshape(num_rows, num_columns).T  # 轉置以符合每列顯示的要求

    plt.rcParams['font.family'] = 'Microsoft JhengHei' # 設置中文字體
    plt.figure(figsize = (12, 6), facecolor = 'black')
    plt.xlabel("交易日（週）", fontsize = 11, color = 'white') # x 軸的標籤
    plt.title('模型預測分布情形', fontsize = 14, color = 'white', 
              va = 'baseline')
     
    # 設定配色
    colors = ['#636363', '#00EB00', '#9c9c9c'] # 黑、綠、灰
    cmap = mcolors.ListedColormap(colors) # 自訂顏色映射:補齊、0(對) 、1(錯)
    bounds = [-1, 0, 1, 2] # -1：補齊數據, 0：原始數據, 1：補齊數據
# =============================================================================
#     bounds 的 2 是用來定義顏色映射的邊界範圍，並且它不會直接影響圖像中的顏色。
# =============================================================================
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
       
    # 繪製圖像
    plt.imshow(result_2d, cmap = cmap, norm = norm, interpolation = 'none', 
               aspect = 'equal')
    
    # 繪製每個方格的邊框
    ax = plt.gca()
    num_rows, num_columns = result_2d.shape
    
    for i in range(num_rows):
        for j in range(num_columns):
            rect = patches.Rectangle((j - 0.5, i - 0.55), 1, 1, linewidth = 1, 
                                     edgecolor = '#636363', 
                                     facecolor = 'none')
            ax.add_patch(rect)
            
    # 設置圖表整體邊框顏色和寬度
    for spine in ax.spines.values():
        spine.set_edgecolor('#636363')
        spine.set_linewidth(1)
    
    # 設置刻度字體大小
    plt.xticks([i for i in range(0, len(result) // 5, 5)], 
               [i if i != 0 else 1 for i in range(0, len(result) // 5, 5)], 
               fontsize = 10, color = 'white')
    plt.yticks(range(5), ['1st', '2nd', '3rd', '4th', '5th'], 
               fontsize = 9, color = 'white', ha = 'left')
    plt.tick_params(axis = 'y', pad = 16) # 因 ha = 'left' 時會導致重疊顯示，需校正
# =============================================================================
#     axis：指定軸
#     pad：偏移量，正值左移參考軸，負值右移。pad 是刻度標籤與刻度線之間的距離
# =============================================================================
    
    # 創建圖例框
    legend_labels = ['自動補齊', '答對', '答錯']
    legend_elements = [plt.Line2D([], [], linestyle = '-.', 
                                  color = color, 
                                  lw = 6, 
                                  label = label) for color, 
                       label in zip(colors[1:], legend_labels[1:])]
# =============================================================================
#     plt.Line2D：
#     
#     plt.Line2D([0], [0], color = color, lw = 4, label = label) 創建了一個線條對象，用於圖例。
#     linestyle:'-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 
#     'dashdot', 'dotted'
#     color = color 設定線條顏色。
#     lw = 4 設定線條寬度。
#     label = label 設定圖例標籤。
# =============================================================================
    
    # 添加圖例
    plt.legend(handles = legend_elements, loc = 'upper left',
               ncol = 3, bbox_to_anchor = (-0.01, 1.55), 
               facecolor = 'black', labelcolor = 'w')
# =============================================================================
#     loc：圖例於圖表中的位置
#     ncol：單一行顯示幾個標籤
#     bbox_to_anchor：以 loc 為原點，進階調整圖例位置
#     facecolor：圖例背景色   
#     labelcolor：圖例統一文字顏色
# =============================================================================
    

# 將 numpy.ndarray 接回成 DataFrame
train_df = pd.DataFrame(trainX, columns = feature_names)
test_df = pd.DataFrame(testX, columns = feature_names)
test_df['xor_result'] = xor_result
#print("Train DataFrame:\n", train_df.head())
#print("\nTest DataFrame:\n", test_df.head())

# 使用 'Index' 進行回朔資料
test_df = test_df[['Index', 'Open', 'xor_result']].merge(
    df[['DATE', 'Index', 'High', 'Low', 'Close']], on = 'Index', how = 'left')
test_df.set_index('DATE', inplace = True)
test_df = test_df.sort_values(by = 'DATE', ascending = False) # 依時間排序
#print("test_df type: ", type(test_df))
#print("test_df len: ", len(test_df))
#print('test_df columns：\n', test_df.columns)
#print("test_df head:\n", test_df.head())
#print("test_df tail:\n", test_df.tail())

# 繪製熱量圖
heatmap_darw(test_df['xor_result'])

# 設定繪圖參數
KLine_df = test_df.resample('W').agg({'Open': 'first', 'High': 'max', 
                                      'Low': 'min', 'Close': 'last'})
# 綠漲紅跌
KLine_df['Color'] = KLine_df.apply(lambda row: 'g' if row['Close'] > 
                                   row['Open'] else 'r', axis = 1)

# 繪製 K 線圖
plt.figure(figsize = (12, 6))

# 繪製 K 棒
for i in range(len(KLine_df)):
    row = KLine_df.iloc[i]
    color = row['Color']
    plt.plot([KLine_df.index[i], KLine_df.index[i]], 
             [row['Low'], row['High']], color = color, linewidth = 1)  # 垂直線
    plt.plot([KLine_df.index[i], KLine_df.index[i]], 
             [row['Open'], row['Close']], color = color, linewidth = 5)  # K 棒

plt.xlabel('週')
plt.ylabel('價格')
plt.title('週 K 線圖')
plt.xticks(rotation = 45)
plt.grid(True)
