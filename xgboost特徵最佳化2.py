# =============================================================================
# 相較於前版，2 嘗試臨時新增欄位 Index 供資料回朔，並新增週 K 與熱力圖的對照圖
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

print('Xgboost測試集準確率 %.3f' % test_acc)
print(f"訓練時間: {training_time // 60:.0f} 分 {training_time % 60:.2f} 秒")
# 0.821


# 這裡本圖僅參考分布，不具實際應用意義，因 train_test_split 抽選資料為隨機抽取(並
# 不連續)，並且輸出的資料完全不可回朔，僅能自行添加欄位回朔，不利於本繪圖的時間資料。
import matplotlib.colors as mcolors
import matplotlib.patches as patches


def heatmap_darw(testX, feature_names):
    # 將 numpy.ndarray 接回成 DataFrame
    test_df = pd.DataFrame(testX, columns = feature_names)
    test_df['xor_result'] = xor_result

    # 使用 'Index' 進行回朔資料
    test_df = test_df[['Index', 'Open', 'xor_result']].merge(
        df[['DATE', 'Index', 'High', 'Low', 'Close']], on = 'Index', 
        how = 'left')
    test_df.set_index('DATE', inplace = True)
    test_df = test_df.sort_values(by = 'DATE', ascending = False) # 依時間排序

    # 定義函數計算每個分組的聚合值
    def aggregate_group(group):
        return pd.Series({
            'Open': group['Open'].iloc[0],  # 第一筆 Open 值
            'High': group['High'].max(),    # 最大 High 值
            'Low': group['Low'].min(),      # 最小 Low 值
            'Close': group['Close'].iloc[-1] # 最後一筆 Close 值
        })

    
    # 生成分組標籤
    group_labels = np.arange(len(test_df) // 5 + (1 if len(test_df) % 5 > 0 else 0))
    group_labels = np.repeat(group_labels, 5)[:len(test_df)]  # 重複標籤並修正長度

    test_df['Group'] = group_labels

    # 根據分組進行聚合
    KLine_df = test_df.groupby(by = 'Group').apply(
        aggregate_group, include_groups = False).reset_index(drop = True)
# =============================================================================
#     # DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
#     KLine_df = test_df.groupby(by = 'Group').apply(
#         aggregate_group).reset_index(drop = True)
# =============================================================================
    
    # 設定綠漲紅跌
    KLine_df['Color'] = KLine_df.apply(lambda row: 'g' if row['Close'] > 
                                       row['Open'] else 'r', axis = 1)

    # 繪製 K 線圖
    plt.figure(figsize = (12, 6), facecolor = 'black')
    plt.subplot(2, 1, 1).set_facecolor('black') # 設該子圖背景為黑色
    # =========================================================================
    # plt.subplot(nrows, ncols, index)
    # nrows：子圖的列數，垂直向要切成幾張子圖。
    # ncols：子圖的欄數，水平向要切成幾張子圖。
    # index：子圖索引，從 1 開始，由左至右的順序排列。
    # =========================================================================

    # 繪製 K 棒
    for i in range(len(KLine_df)):
        row = KLine_df.iloc[i]
        color = row['Color']
        plt.plot([KLine_df.index[i], KLine_df.index[i]], 
                 [row['Low'], row['High']], color = color, linewidth = 1) # 垂直細線
        plt.plot([KLine_df.index[i], KLine_df.index[i]], 
                 [row['Open'], row['Close']], color = color, linewidth = 5) # 垂直粗線

    plt.xlim(-0.5, len(KLine_df) - 0.5) # 調整距離，與下圖對應
    plt.gca().axes.get_xaxis().set_visible(False) # 隱藏 x 軸
    plt.yticks(fontsize = 10, color = 'white')
    plt.ylabel('價格', fontsize = 11, color = 'white')
    plt.title('【週 K 線】 與 【模型預測分布】 對照圖', fontsize = 14, 
              color = 'white', va = 'baseline')    
    plt.grid(True)
    
    # 添加上圖例（但參考依據是下子圖物件）
    colors = ['g', 'r']
    legend_labels = ['上漲', '下跌']
    legend_elements = [plt.Line2D([], [], linestyle = '-.', 
                                  color = colors, 
                                  lw = 6, 
                                  label = label) for colors, 
                       label in zip(colors, legend_labels)]
    plt.legend(handles = legend_elements, loc = 'upper left',
               ncol = 2, bbox_to_anchor = (-0.01, 1.13), 
               facecolor = 'black', labelcolor = 'w')
    
    
    # 繪製熱量圖
    result = test_df['xor_result']
    
    num_columns = 5 # 每列 5 筆數據
    num_rows = (len(result) + num_columns - 1) // num_columns # 計算需要的行數
    data_padded = np.pad(result, 
                         (0, num_rows * num_columns - len(result)), 
                         mode = 'constant', constant_values = -1) # 補齊數據
    
    # 重塑為每列 5 筆數據的二維矩陣
    result_2d = data_padded.reshape(num_rows, num_columns).T  # 轉置以符合每列顯示的要求
    
    plt.rcParams['font.family'] = 'Microsoft JhengHei' # 設置中文字體
    plt.subplot(2, 1, 2)
    # =========================================================================
    # plt.subplot(nrows, ncols, index)
    # nrows：子圖的列數，垂直向要切成幾張子圖。
    # ncols：子圖的欄數，水平向要切成幾張子圖。
    # index：子圖索引，從 1 開始，由左至右的順序排列。
    # =========================================================================
     
    # 設定配色
    colors = ['#636363', '#00EB00', '#9c9c9c'] # 黑、綠、灰
    cmap = mcolors.ListedColormap(colors) # 自訂顏色映射:補齊、0(對) 、1(錯)
    bounds = [-1, 0, 1, 2] # -1：補齊數據, 0：原始數據, 1：補齊數據
# =============================================================================
#     bounds 的 2 是用來定義顏色映射的邊界範圍，並且它不會直接影響圖像中的顏色。
# =============================================================================
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
       
    # 繪圖
    plt.imshow(result_2d, cmap = cmap, norm = norm, interpolation = 'none', 
               aspect = 'equal')
    
    # 加繪每個方格的邊框
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
    plt.xlabel("交易日（週）", fontsize = 11, color = 'white') # x 軸的標籤
    plt.ylabel("每組交易日次序", fontsize = 11, color = 'white') # x 軸的標籤
    
    plt.tick_params(axis = 'y', pad = 16) # 因 ha = 'left' 時會導致重疊顯示，需校正
# =============================================================================
#     axis：指定軸
#     pad：偏移量，正值左移參考軸，負值右移。pad 是刻度標籤與刻度線之間的距離
# =============================================================================
    
    # 創建圖例框
    legend_labels = ['自動補齊', '正確預測', '錯誤預測']
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
    plt.legend(handles = legend_elements, loc = 'lower left',
               ncol = 3, bbox_to_anchor = (-0.005, -0.9), 
               facecolor = 'black', labelcolor = 'w')
# =============================================================================
#     loc：圖例於圖表中的位置
#     ncol：單一行顯示幾個標籤
#     bbox_to_anchor：以 loc 為原點，進階調整圖例位置
#     facecolor：圖例背景色   
#     labelcolor：圖例統一文字顏色
# =============================================================================
    
    plt.subplots_adjust(hspace = -0.3)  # 調整子圖之間的垂直間距
    
heatmap_darw(testX, feature_names)
