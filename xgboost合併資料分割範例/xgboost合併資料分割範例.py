# =============================================================================
# 分批次個別訓練模型
# =============================================================================

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import math
from xgboost import XGBClassifier
import time
import matplotlib.pyplot as plt

# 讀取資料
df = pd.read_excel("../data.xlsx")
print(f"總資料數：{len(df)}")

df_Latest = pd.read_excel("../data_Latest.xlsx")
print(f"最新數據測試集總資料數：{len(df_Latest)}")

feature_names = ['Gold_Close', 'Gold_High', 'CPIAUCNS', 'Gold_Open', 'UNRATE', 
                 'MA_20', 'MA_10', 'USD_Index_Growth_Rate', 'TW_CPI_Rate', 
                 'WILLR', 'Open', 'K', 'RSI_14', 'Gold_Volume', 
                 'Gold_Growth_Rate', 'FEDFUNDS', 'Bollinger Bands lower', 
                 'Bollinger Bands Upper', 'USA_GDP_Rate']

label_column = 'LABEL'

X = df[feature_names].values
y = df[label_column].values

X_Latest = df_Latest[feature_names].values
y_Latest = df_Latest[label_column].values


n_splits = 24 # 設定分割數量
overlap_rate = 3 * 0.1 + 1 # 設定每批訓練集之間的，最低重疊率
max_train_size = math.ceil(len(df) // n_splits * overlap_rate) if \
    len(df) % n_splits == 0 else math.ceil(len(df) / n_splits * overlap_rate) # 計算最大訓練集大小
print(f"最大資料長度 {max_train_size}")

# 初始化 TimeSeriesSplit
TSS = TimeSeriesSplit(n_splits = n_splits, max_train_size = max_train_size)
# 不可指定訓練與測試集的占比

# 儲存每個折的訓練集索引
split_indices = [None for _ in range(n_splits)]
labels = df['LABEL']
batch_test_scores = []
latest_test_scores = []

# 逐步分割資料
for i, (train_index, test_index) in enumerate(TSS.split(X, y)):
    split_indices[i] = train_index
    sub_train_index = df.iloc[train_index]
    sub_test_index = df.iloc[test_index]
    train_len = len(sub_train_index)
    test_len = len(sub_test_index)
    all_len = train_len + test_len
    
    print('--------------')
    print(f"第 {i+1} 份")
    print(f"子訓練集資料數：{train_len}")
    print(f"子測試集資料數：{test_len} (佔批次 {test_len / all_len * 100:.1f} %)")
    print(f"子訓練集索引（頭尾 5 個）：{sub_train_index.index.tolist()[:5]}...{sub_train_index.index.tolist()[-5:]}")
    print(f"子測試集索引（頭尾 5 個）：{sub_test_index.index.tolist()[:5]}...{sub_test_index.index.tolist()[-5:]}")
    
    # 提取訓練集和測試集
    sub_trainX, sub_testX = X[train_index], X[test_index]
    sub_trainY, sub_testY = y[train_index], y[test_index]
    
    # 訓練 XGBoost 模型
    Xgboost = XGBClassifier()
    start_time = time.time()
    Xgboost.fit(sub_trainX, sub_trainY)
    training_time = time.time() - start_time
    
    # 批次測試集預測結果和模型準確率
    test_acc = Xgboost.score(sub_testX, sub_testY)
    batch_test_scores.append(test_acc)
      
    print('Xgboost測試集準確率 %.3f' % test_acc)
    print(f"訓練時間: {training_time // 60:.0f} 分 {training_time % 60:.2f} 秒")
    
    # 最新資料測試
    latest_data_test_acc = Xgboost.score(X_Latest, y_Latest)
    latest_test_scores.append(latest_data_test_acc)


# 計算重疊率
def calculate_overlap_rate(indices_list):
    total_overlap = 0
    total_count = len(df)
    
    for i in range(len(indices_list)):
        for j in range(i + 1, len(indices_list)):
            overlap_count = len(set(indices_list[i]).intersection(set(indices_list[j])))
            total_overlap += overlap_count

    overlap_rate = total_overlap / total_count
    return overlap_rate

overlap_rate = calculate_overlap_rate(split_indices)
print(f"資料彼此重疊率：{overlap_rate * 100:.2f} %")

# 顯示每批準確度變化
plt.rcParams['font.family'] = 'Microsoft JhengHei' # 設置中文字體

# 設置背景顏色
plt.gcf().patch.set_facecolor('black')  # 設置整個圖表背景為黑色
plt.gca().set_facecolor('black')  # 設置坐標軸背景為黑色

# =============================================================================
# fig, ax = plt.subplots()
# fig.patch.set_facecolor('black')  # 設置整個圖表背景為黑色
# ax.set_facecolor('black')  # 設置坐標軸背景為黑色
# =============================================================================

plt.plot(range(1, len(batch_test_scores) + 1), batch_test_scores, 
         label = '批次測試集準確率', color = 'blue')
plt.plot(range(1, len(batch_test_scores) + 1), latest_test_scores, 
         label = '最新數據準確率', color = 'lime')
plt.xticks([i for i in range(1, len(batch_test_scores) + 1)], color = 'white')
plt.xlabel("序位（批）", color = 'white')
ytick = [i / 100 for i in range(0, 105, 10)]
plt.yticks(ytick, [str(int(i * 100)) + ' %' for i in ytick], color = 'white')
plt.ylabel("準確率", color = 'white')
plt.title("分批個別訓練 - 準確率", color = 'white') 
plt.grid(True, axis = 'y')
plt.legend(loc = 'lower left', facecolor = 'black', labelcolor = 'w')