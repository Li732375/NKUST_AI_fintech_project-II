# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:54:18 2024

@author: 郭昱
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNNmodel
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

df = pd.read_excel('外匯data.xlsx',index_col = 'Date')

def split_stock_data(stock_data, label_column, delete_column, test_size = 0.3, 
                     random_state = 42):
    X = stock_data.drop([label_column, *delete_column], 
                        axis = 1).values # X為特徵，刪除非特徵的欄位
    y = stock_data[label_column].values # y為標籤(LABEL)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = test_size, 
                                                        random_state = 
                                                        random_state) # 資料分割
    
    feature_names = [f'{i}' for i in stock_data.drop([label_column, 
                                                      *delete_column], 
                                                     axis = 1)]

    return X_train, X_test, y_train, y_test, feature_names

label_column = 'LABEL' # 標籤欄位
delete_column = ['Next_5Day_Return', 'Volume'] # 刪除的欄位
trainX, testX, trainY, testY, feature_names = split_stock_data(df, label_column, 
                                                delete_column)

model_accuracies = {}

import time

'''
KNN = KNNmodel() # 設定KNN模型
start_time = time.time()
KNN.fit(trainX, trainY) # 模型訓練
training_time = time.time() - start_time

train_acc = KNN.score(trainX, trainY) #訓練集準確度計算
test_acc = KNN.score(testX, testY) #測試集準確度計算
model_accuracies['KNN'] = test_acc #將測試集結果儲存到字典中
print('KNN訓練集準確率 %.2f' % train_acc) # 輸出訓練集的準確度
print('KNN驗證集準確率 %.2f' % test_acc) # 輸出驗證集的
print(f"測試時間: {training_time:.8f} 秒")


SVM = svm.SVC()
start_time = time.time()
SVM.fit(trainX, trainY)
training_time = time.time() - start_time

train_acc = SVM.score(trainX, trainY)
test_acc = SVM.score(testX, testY)
model_accuracies['SVM'] = test_acc
print(f'SVM訓練集準確率 {train_acc:.2f}')
print(f'SVM驗證集準確率 {test_acc:.2f}')
print(f"測試時間: {training_time:.8f} 秒")


Logistic = LogisticRegression()
start_time = time.time()
Logistic.fit(trainX, trainY)
training_time = time.time() - start_time

train_acc = Logistic.score(trainX, trainY)
test_acc = Logistic.score(testX, testY)
model_accuracies['LogisticRegression'] = test_acc
print(f'LR訓練集準確率 {train_acc:.2f}')
print(f'LR測試集準確率 {test_acc:.2f}')
print(f"測試時間: {training_time:.8f} 秒")


Bayesian = GaussianNB()
start_time = time.time()
Bayesian.fit(trainX, trainY)
training_time = time.time() - start_time

train_acc = Bayesian.score(trainX, trainY)
test_acc = Bayesian.score(testX, testY)
model_accuracies['GaussianNB'] = test_acc
print(f'Bayes訓練集準確率 {train_acc:.2f}')
print(f'Bayes測試集準確率 {test_acc:.2f}')
print(f"測試時間: {training_time:.8f} 秒")


RF = RandomForestClassifier()
start_time = time.time()
RF.fit(trainX, trainY)
training_time = time.time() - start_time

train_acc = RF.score(trainX, trainY)
test_acc = RF.score(testX, testY)
model_accuracies['RandomForest'] = test_acc
print(f'RF訓練集準確率 {train_acc:.2f}')
print(f'RF測試集準確率 {test_acc:.2f}')
print(f"測試時間: {training_time:.8f} 秒")
'''

Xgboost = XGBClassifier()
start_time = time.time()
Xgboost.fit(trainX, trainY)
training_time = time.time() - start_time

train_acc = Xgboost.score(trainX, trainY)
test_acc = Xgboost.score(testX, testY)
model_accuracies['XGBoost'] = test_acc
print('Xgboost訓練集準確率 %.2f' % train_acc)
print('Xgboost測試集準確率 %.2f' % test_acc)
print(f"測試時間: {training_time:.8f} 秒")


# 繪製特徵重要性圖
import matplotlib.pyplot as plt

# 將特徵名稱和重要性配對
feature_importance_pairs = list(zip(feature_names, 
                                    Xgboost.feature_importances_))

sorted_pairs = sorted(feature_importance_pairs, key = lambda x: x[1], 
                      reverse = True)

# 提取排序後的特徵，[:] 取得前幾名的特徵和重要性
sorted_feature_names, sorted_importances = zip(*sorted_pairs[:25])
print(sorted_feature_names)

# 繪製特徵重要性橫條圖
plt.rcParams['font.family'] = 'Microsoft JhengHei' # 設置中文字體
plt.figure(figsize = (12, 8))
bars = plt.barh(sorted_feature_names, sorted_importances, color = 'skyblue')
        
# 顯示每個橫條的數值
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.002, bar.get_y() + bar.get_height()/2, 
             f'{width * 100:.2f} %', 
             va = 'center', ha = 'left', fontsize = 10)
    
plt.xlabel('特徵重要性')
plt.ylabel('特徵')
plt.title('特徵重要性')
plt.tight_layout(pad = 0.5)
plt.gca().invert_yaxis()  # 反轉 y 軸，使重要性高的特徵顯示在上面
plt.show()

best_model = max(model_accuracies, key = model_accuracies.get)
best_accuracy = model_accuracies[best_model]
print(f'準確率最高的模型是 {best_model}，準確率為 %.4f' % best_accuracy)
