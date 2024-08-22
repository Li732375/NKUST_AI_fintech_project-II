import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

df = pd.read_excel('data.xlsx', index_col = 'Date')
#print(df.columns)

def split_stock_data(stock_data, label_column, delete_column, test_size = 0.3, 
                     random_state = 42):
    
    X = stock_data.drop(delete_column, axis = 1)
    feature_names = X.columns.tolist()
    y = stock_data[label_column].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = test_size, 
                                                        random_state = 
                                                        random_state) # 資料分割

    return X_train, X_test, y_train, y_test, feature_names

label_column = 'LABEL' # 標籤欄位
# 刪除的欄位，欄位 'Next_?Day_Return' 中，? 應視特徵處理檔案參數 pre_day 調整
delete_column = ['LABEL', 'Volume_x', 'Next_5Day_Return']

# 切分資料為訓練與測試集
trainX, testX, trainY, testY, feature_names = split_stock_data(df, label_column, 
                                                delete_column)


import time

Xgboost = XGBClassifier()
start_time = time.time()
Xgboost.fit(trainX, trainY)
training_time = time.time() - start_time

test_predic = Xgboost.predict(testX) # 取得預測的結果
test_acc = Xgboost.score(testX, testY)
print('Xgboost測試集準確率 %.2f' % test_acc)


# 繪製特徵重要性圖
import matplotlib.pyplot as plt

# 將特徵名稱和重要性配對
feature_importance_pairs = list(zip(feature_names, 
                                    Xgboost.feature_importances_))

sorted_pairs = sorted(feature_importance_pairs, key = lambda x: x[1], 
                      reverse = True)

# 提取排序後的特徵
sorted_feature_names, sorted_importances = zip(*sorted_pairs[:]) # [:數字] 取得前幾名的特徵和重要性
print("依特徵重要性排序")
print(sorted_feature_names)

# 繪製特徵重要性橫條圖
plt.rcParams['font.family'] = 'Microsoft JhengHei' # 設置中文字體
plt.figure(figsize = (12, 8))
bars = plt.barh(sorted_feature_names, sorted_importances, color = 'skyblue')
        
# 顯示每個橫條的數值
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.002, bar.get_y() + bar.get_height() / 2, 
             f'{width * 100:.2f} %', 
             va = 'center', ha = 'left', fontsize = 10)
    
plt.xlabel('特徵重要性')
plt.ylabel('特徵')
plt.title('特徵重要性')
plt.tight_layout(pad = 0.5)
plt.gca().invert_yaxis()  # 反轉 y 軸，使重要性高的特徵顯示在上面
plt.show()


print(f"測試時間: {training_time:.2f} 秒")
print(f'模型準確率為 {test_acc:.3f}')