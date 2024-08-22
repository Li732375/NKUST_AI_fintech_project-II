import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import math
from xgboost import XGBClassifier
import time
import matplotlib.pyplot as plt

# 讀取資料
df = pd.read_excel("data.xlsx")
print(f"總資料數：{len(df)}")

feature_names = ['Close_y', 'High_y', 'CPIAUCNS', 'Open_y', 'UNRATE', 'MA_20', 
                 'MA_10', 'Growth Rate_x', 'TW_CPI_Rate', 
                 'WILLR', 'Open_x', 'K', 'RSI_14', 'Volume_y', 
                 'Growth Rate_y', 'FEDFUNDS', 'Bollinger Bands lower', 
                 'Bollinger Bands Upper', 'USA_GDP_Rate']

label_column = 'LABEL'

X = df[feature_names].values
y = df[label_column].values

accuracies = []

n_splits = 4 # 設定分割數量
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
test_scores = []

# 逐步分割資料
for i, (train_index, test_index) in enumerate(TSS.split(X, y)):
    split_indices[i] = train_index
    sub_train_index = df.iloc[train_index]
    sub_test_index = df.iloc[test_index]
    train_len = len(sub_train_index)
    test_len = len(sub_test_index)
    all_len = train_len + test_len
    
    print(f"第 {i+1} 份")
    print(f"子訓練集資料數：{train_len}")
    print(f"子測試集資料數：{test_len} (佔該份 {test_len / all_len * 100:.1f} %)")
    print(f"子訓練集索引（頭尾 5 個）：{sub_train_index.index.tolist()[:5]}...{sub_train_index.index.tolist()[-5:]}")
    print(f"子測試集索引（頭尾 5 個）：{sub_test_index.index.tolist()[:5]}...{sub_test_index.index.tolist()[-5:]}")
    print('--------------')
    
    # 提取訓練集和測試集
    sub_trainX, sub_testX = X[train_index], X[test_index]
    sub_trainY, sub_testY = y[train_index], y[test_index]
    
    # 訓練 XGBoost 模型
    Xgboost = XGBClassifier()
    start_time = time.time()
    #print(sub_test)
    Xgboost.fit(sub_trainX, sub_trainY)
    training_time = time.time() - start_time
    
    # 預測結果和模型準確率
    #test_predic = Xgboost.predict(testX)
    #test_acc = Xgboost.score(testX, testY)
    test_acc = Xgboost.score(sub_testX, sub_testY)
    test_scores.append(test_acc)
    
    print('Xgboost測試集準確率 %.3f' % test_acc)
    print(f"訓練時間: {training_time // 60:.2f} 分 {training_time % 60:.2f} 秒")



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

# 顯示數據
plt.rcParams['font.family'] = 'Microsoft JhengHei' # 設置中文字體
plt.plot(range(1, len(test_scores) + 1), test_scores, marker = 'o', label = '測試集準確率')
plt.xlabel("Date")
plt.ylabel("準確率")
plt.title("分批訓練準確率")
plt.legend()