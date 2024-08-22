import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import time
import matplotlib.pyplot as plt
from itertools import combinations

# 讀取數據
df = pd.read_excel('外匯data.xlsx', index_col = 'Date')

# 分割資料
def split_stock_data(stock_data, feature_names, label_column, test_size = 0.3, 
                     random_state = 42):
    X = stock_data[feature_names]
    y = stock_data[label_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = test_size, 
                                                        random_state = 
                                                        random_state)
    return X_train, X_test, y_train, y_test

# 特徵名單
feature_names = ['Bollinger Bands lower_5', 'MA_20_20', 'MA_5_5', 
                 'Bollinger Bands lower', 'MA_10_20', 'RSI_14', 'MACD_20']

label_column = 'LABEL'

# 初始化 XGBoost 模型
model = XGBClassifier()

# 初始化準確率字典
model_accuracies = {}

# 生成所有特徵組合
def get_feature_combinations(features, min_features = 1, max_features=None):
    if max_features is None:
        max_features = len(features)
    feature_combinations = []
    for i in range(min_features, max_features + 1):
        feature_combinations.extend(combinations(features, i))
    return feature_combinations

# 對每一組特徵進行測試
for feature_set in get_feature_combinations(feature_names, 1, len(feature_names)):
    feature_set = list(feature_set)  # 轉換為列表
    print(f'正在測試特徵組合: {feature_set}')
    
    X_train, X_test, y_train, y_test = split_stock_data(df, feature_set, 
                                                        label_column)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    test_acc = model.score(X_test, y_test)
    model_accuracies[str(feature_set)] = test_acc
    
    print(f'特徵組合: {feature_set} 測試集準確率: {test_acc:.4f}')
    print(f"測試時間: {training_time:.4f} 秒")

# 找到最佳的特徵組合
best_feature_set = max(model_accuracies, key = model_accuracies.get)
best_accuracy = model_accuracies[best_feature_set]

print(f'準確率最高的特徵組合是: {best_feature_set}')
print(f'準確率為: {best_accuracy:.4f}')

# 繪製特徵重要性圖
# 使用最佳特徵組合重新訓練模型
X_train, X_test, y_train, y_test = split_stock_data(df, eval(best_feature_set), 
                                                    label_column)
model.fit(X_train, y_train)

# 獲取特徵重要性並繪製圖表
feature_importance_pairs = list(zip(eval(best_feature_set), 
                                    model.feature_importances_))
sorted_pairs = sorted(feature_importance_pairs, key = lambda x: x[1], 
                      reverse=True)

sorted_feature_names, sorted_importances = zip(*sorted_pairs)

# 繪製特徵重要性橫條圖
plt.figure(figsize=(12, 8))
bars = plt.barh(sorted_feature_names, sorted_importances, color = 'skyblue')

# 顯示每個橫條的數值
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.002, bar.get_y() + bar.get_height() / 2, 
             f'{width * 100:.2f} %', va = 'center', ha = 'left', fontsize = 10)

plt.xlabel('特徵重要性')
plt.ylabel('特徵')
plt.title('特徵重要性')
plt.tight_layout(pad=0.5)
plt.gca().invert_yaxis()  # 反轉 y 軸，使重要性高的特徵顯示在上面
plt.show()

