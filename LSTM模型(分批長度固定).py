import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import time
import matplotlib.pyplot as plt

# 加載數據集
data = pd.read_excel('data.xlsx')

feature_names = ['Close_y', 'High_y', 'CPIAUCNS', 'Open_y', 'UNRATE', 'MA_20', 
                 'MA_10', 'Growth Rate_x', 'USA_CPI_Rate', 'TW_CPI_Rate', 
                 'WILLR', 'Open_x', 'K', 'RSI_14', 'Volume_y', 
                 'Growth Rate_y', 'FEDFUNDS', 'Bollinger Bands lower', 
                 'Bollinger Bands Upper', 'USA_GDP_Rate']
# 特徵選擇，去掉不需要的欄位
#X = data.drop(['LABEL', 'Date', 'Volume_x', 'Next_5Day_Return'], axis = 1).values

X = data[feature_names].values
y = data['LABEL'].values

# 數據縮放
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# 轉換數據格式為 3D (樣本數，時間步，特徵數)
time_steps = 10  # 假設每個樣本有 10 個時間步
X_reshaped = []
y_reshaped = []

for i in range(len(X_scaled) - time_steps):
    X_reshaped.append(X_scaled[i:i + time_steps])
    y_reshaped.append(y[i + time_steps])

X_reshaped = np.array(X_reshaped)
y_reshaped = np.array(y_reshaped)

# 使用 TimeSeriesSplit 進行分割
tscv = TimeSeriesSplit(n_splits = 5)

units_1 = 256
adam_learning_rate = 0.001
epochs = 50
batch_size = 16

train_accuracies = []
test_accuracies = []

for fold, (train_index, test_index) in enumerate(tscv.split(X_reshaped)):
    X_train, X_test = X_reshaped[train_index], X_reshaped[test_index]
    Y_train, Y_test = y_reshaped[train_index], y_reshaped[test_index]

    # 構建 LSTM 模型
    model = Sequential()
    model.add(LSTM(units = units_1, return_sequences = False, 
                   input_shape = (X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1, activation = 'sigmoid'))  # 使用 sigmoid 激活函數進行二元分類
    model.compile(optimizer = Adam(learning_rate = adam_learning_rate), 
                  loss = 'mean_squared_error', metrics = ['accuracy'])  # 編譯模型

    start_time = time.time()

    # 訓練模型
    model.fit(X_train, Y_train, epochs = epochs, batch_size = batch_size, 
              validation_data = (X_test, Y_test))

    training_time = time.time() - start_time

    # 訓練集和測試集準確率
    train_loss, train_acc = model.evaluate(X_train, Y_train, verbose = 0)  
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose = 0)  
    
    # 紀錄每一折的準確率
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    
    print(f'Fold {fold + 1} - LSTM 訓練集準確率: %.3f' % train_acc)
    print(f'Fold {fold + 1} - LSTM 測試集準確率: %.3f' % test_acc)
    print(f"Fold {fold + 1} 訓練時間: {training_time // 60:.2f} 分 {training_time % 60:.2f} 秒")

# 繪製準確率變化圖
plt.rcParams['font.family'] = 'Microsoft JhengHei'  # 設置中文字體
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, marker = 'o', label = '訓練集準確率')
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, marker = 'o', label = '測試集準確率')
plt.xlabel("折數")
plt.ylabel("準確率")
plt.title("LSTM 模型每一折的準確率變化")
plt.legend()
