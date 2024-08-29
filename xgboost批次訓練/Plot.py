import matplotlib.pyplot as plt

def AccLineAndDataArea_Draw(batch_test_scores, latest_test_scores, TSSsplit, 
                            AllData_Len, n_splits):
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
    
    # =============================================================================
    # 訓練集和測試集區間的橫條圖
    plt.figure(figsize = (12, 6))
    plt.gcf().patch.set_facecolor('black')  # 設置整個圖表背景為黑色
    plt.gca().set_facecolor('black')  # 設置坐標軸背景為黑色
    
    # 繪製橫條圖
    for i, (train_index, test_index) in enumerate(TSSsplit):
        train_start = min(train_index)
        train_end = max(train_index)
        test_start = min(test_index)
        test_end = max(test_index)
        
        plt.barh(i + 1, train_end - train_start + 1, left = train_start, 
                 color = 'blue', edgecolor = 'white', 
                 label = '訓練集' if i == 0 else "")
        plt.barh(i + 1, test_end - test_start + 1, left = test_start, 
                 color = 'red', edgecolor = 'white', 
                 label = '測試集' if i == 0 else "")
    
    plt.xlim(-1, AllData_Len + 0.5)
    plt.xticks(color = 'white')
    plt.xlabel("資料索引", color = 'white', fontsize = 11)
    plt.ylim(1, n_splits + 1)
    plt.yticks(range(0, n_splits + 1, 5), [''] + [str(i) for i in range(5, n_splits + 1, 5)], 
               color = 'white')
    plt.ylabel("批次", color = 'white', fontsize = 11)
    plt.title("批次區間", color = 'white', fontsize = 14)
    plt.legend(loc = 'lower right', facecolor = 'black', fontsize = 11, 
               labelcolor = 'white')
    plt.grid(True, axis = 'x', color = 'white', linewidth = 0.5)