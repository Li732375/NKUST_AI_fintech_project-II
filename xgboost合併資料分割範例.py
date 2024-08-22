import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import math
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import time

# 讀取資料
df = pd.read_excel("data.xlsx", index_col='Date')
print(df.columns)

# 計算資料分布比例
label_proportions = df['LABEL'].value_counts(normalize=True)
print(f"總資料數：{len(df)}")
print("全部資料分布：")
print(label_proportions)
print('--------------')



def split_stock_data(stock_data, label_column, delete_column, test_size = 0.3, 
                     random_state = 42):
    X = stock_data[feature_names].values
    y = stock_data[label_column].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = test_size, 
                                                        random_state = 
                                                        random_state)

    return X_train, X_test, y_train, y_test, feature_names

feature_names = ['Close_y', 'High_y', 'CPIAUCNS', 'Open_y', 'UNRATE', 'MA_20', 
                 'MA_10', 'Growth Rate_x', 'TW_CPI_Rate', 
                 'WILLR', 'Open_x', 'K', 'RSI_14', 'Volume_y', 
                 'Growth Rate_y', 'FEDFUNDS', 'Bollinger Bands lower', 
                 'Bollinger Bands Upper', 'USA_GDP_Rate']

label_column = 'LABEL'
delete_column = ['LABEL', 'Volume_x', 'Next_5Day_Return']

trainX, testX, trainY, testY, feature_names = split_stock_data(df, label_column, 
                                                delete_column)


# 初始化 TimeSeriesSplit 參數
n_splits = 12  # 設定分割數量 
overlap_rate = 3 * 0.1 + 1  # 設定每批訓練集之間的最低重疊率
max_train_size = math.ceil(len(df) // n_splits * overlap_rate) if len(df) % n_splits == 0 else math.ceil(len(df) / n_splits * overlap_rate)
print(f"最大資料長度 {max_train_size}")

# 初始化 TimeSeriesSplit
TSS = TimeSeriesSplit(n_splits = n_splits, max_train_size = max_train_size)

# 儲存每個折的訓練集索引
split_indices = [None for _ in range(n_splits)]
test_scores = []

# 逐步分割資料
for i, (train_index, test_index) in enumerate(TSS.split(trainX, trainY)):
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
    sub_trainX, sub_testX = trainX[train_index], trainX[test_index]
    sub_trainY, sub_testY = trainY[train_index], trainY[test_index]
    
    # 訓練 XGBoost 模型
    Xgboost = XGBClassifier()
    start_time = time.time()
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
