import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import math

# 讀取資料
df = pd.read_excel("data.xlsx")

# 計算資料分布比例
label_proportions = df['LABEL'].value_counts(normalize = True)
print(f"總資料數：{len(df)}")
print("全部資料分布：")
print(label_proportions)
print('--------------')

# 切分資料成訓練集和測試集 (7:3)
train_df, test_df = train_test_split(df, test_size = 0.3, random_state = 42)
# 這裡由於是隨機抽取，對時間序列資料可能不合適
print(f"訓練集資料數：{len(train_df)}")
print(f"測試集資料數：{len(test_df)}")

n_splits = 4 * 11 # 設定分割數量 (年分 * 月數)
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
combined_splits = [] # 儲存每一折的訓練集和測試集

for i, (train_index, test_index) in enumerate(TSS.split(df, labels)):
    split_indices[i] = train_index
    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]
    train_len = len(train_df)
    test_len = len(test_df)
    all_len = train_len + test_len
        
    combined_df = pd.concat([train_df, test_df])  # 合併訓練集和測試集
    combined_splits.append(combined_df)
    
    print(f"第 {i+1} 份")
    print(f"訓練集資料數：{train_len} (佔該份 {train_len / all_len * 100:.1f} %)")
    print(f"測試集資料數：{test_len} (佔該份 {test_len / all_len * 100:.1f} %)")
    print(f"訓練集索引（頭尾 5 個）：{train_df.index.tolist()[:5]}...{train_df.index.tolist()[-5:]}")
    print(f"測試集索引（頭尾 5 個）：{test_df.index.tolist()[:5]}...{test_df.index.tolist()[-5:]}")
    print(f"合併後訓練集索引（頭尾 5 個）：{combined_df.index.tolist()[:5]}...{combined_df.index.tolist()[-5:]}")
    print(combined_df.head())
    print('--------------')

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
print(f"資料1彼此重疊率：{overlap_rate * 100:.2f} %")