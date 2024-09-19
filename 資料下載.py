import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

Data_Time_Start = '2003-01-01' # 應後續特徵處理時避免缺值，必須再提前目標一年
Data_Time_End = '2023-12-31'
DataFolder = 'Ori_Data/'


Currency_symbol = 'TWD%3DX' # 輸入股票代號下載匯率資料
Currency_data = yf.download(Currency_symbol, 
                            start = Data_Time_Start, end = Data_Time_End) # 獲取特定日期範圍的匯率資料

excel_filename = f'USDto{Currency_symbol[:3]}_Currency_Data.xlsx' # 將匯率資料存儲為 Excel 檔案，以匯率代號作為檔案名稱
Currency_data.to_excel(DataFolder + excel_filename)
print(f"匯率資料已儲存至 '{DataFolder + excel_filename}'")
print(Currency_data.head())

# 顯示數據
Currency_data['Close'].plot() # 畫出圖形
plt.xlabel("Date") # x 軸標題
plt.ylabel("Closing Price") # y 軸標題
plt.title("USD -> TWD") # 圖標題
plt.show()


#下載美元指數與其成長率
dxy_data = yf.download("DX-Y.NYB", start = Data_Time_Start, end = Data_Time_End)
dxy_data['Growth Rate'] = dxy_data['Close'].pct_change() # 計算每日的成長率（百分比）
print(dxy_data[['Close', 'Growth Rate']].head())# 輸出結果
dxy_data = dxy_data[['Close', 'Growth Rate']]

excel_filename = 'Dx-y_Data.xlsx'
dxy_data.to_excel(DataFolder + excel_filename)
print("美元指數與成長率已保存至 '{DataFolder + excel_filename}'")

# 顯示數據
dxy_data['Close'].plot() # 畫出圖形
plt.xlabel("Date") # x 軸標題
plt.ylabel("Close") # y 軸標題
plt.title("DX-Y.NYB") # 圖標題
plt.show()


# 下載黃金價格數據
gold_data = yf.download("GC=F", start = Data_Time_Start, end = Data_Time_End)
gold_data['Growth Rate'] = gold_data['Close'].pct_change() # 計算每日的成長率（百分比）
print(gold_data[['Close', 'Growth Rate']].head())# 輸出結果

excel_filename = 'Gold_Data.xlsx'
gold_data.to_excel(DataFolder + excel_filename)
print("黃金價格已保存至 '{DataFolder + excel_filename}'")

# 顯示數據
gold_data['Close'].plot() # 畫出圖形
plt.xlabel("Date") # x 軸的標題
plt.ylabel("Close") # y 軸的標題
plt.title("Gold_data") # 圖標題
plt.show()


import pandas_datareader.data as WebData

# pip install pandas_datareader
# 下載聯邦基金利率數據
fed_funds_rate = WebData.DataReader('FEDFUNDS', 'fred', 
                                    start = Data_Time_Start, 
                                    end = Data_Time_End)

excel_filename = 'Fed_Funds_Rate.xlsx'
fed_funds_rate.to_excel(DataFolder + excel_filename)
print(f"匯率資料已存儲為 '{DataFolder + excel_filename}'")
print(fed_funds_rate.head())

# 顯示數據
fed_funds_rate['FEDFUNDS'].plot() # 畫出圖形
plt.xlabel("Date") # x 軸標題
plt.ylabel("Funds Rate") # y 軸標題
plt.title("Fed Funds Rate") # 圖標題
plt.show()


# 下載美國 CPI 數據
cpi_data = WebData.get_data_fred('CPIAUCNS',
                                       start = Data_Time_Start, 
                                       end = Data_Time_End)
cpi_data['USA_CPI_Rate'] = cpi_data['CPIAUCNS'].pct_change() # 算出增長率

excel_filename = 'USA_CPI_Data.xlsx'
cpi_data.to_excel(DataFolder + excel_filename)
print(f"美國 cpi 資料已儲存至 '{DataFolder + excel_filename}'")
print(cpi_data.head())

# 顯示數據
plt.rcParams['font.family'] = 'Microsoft JhengHei' # 設置中文字體
cpi_data['CPIAUCNS'].plot() # 畫出圖形
plt.xlabel("Date") # x 軸標題
plt.ylabel("USA CPI") # y 軸標題
plt.title("美國 CPI") # 圖標題
plt.show()


# 下載美國失業率數據
unemployment_rate = WebData.get_data_fred('UNRATE',
                                          start = Data_Time_Start, 
                                          end = Data_Time_End)

excel_filename = 'USA_Unemployment_Rate.xlsx'
unemployment_rate.to_excel(DataFolder + excel_filename)
print(f"美國失業率資料已儲存至 '{DataFolder + excel_filename}'")
print(unemployment_rate.head())

# 顯示數據
plt.rcParams['font.family'] = 'Microsoft JhengHei' # 設置中文字體
unemployment_rate['UNRATE'].plot() # 畫出圖形
plt.xlabel("Date") # x 軸的標題
plt.ylabel("USA Unemployment Rate") # y 軸的標題
plt.title("美國失業率") # 圖標題
plt.show()


# 獲取 GDP 資料
gdp_data = WebData.get_data_fred('GDP',
                                 start = Data_Time_Start, 
                                 end = Data_Time_End)
gdp_data['USA_GDP_Rate'] = gdp_data['GDP'].pct_change() # 算出增長率
excel_filename = 'USA_GDP.xlsx'
gdp_data.to_excel(DataFolder + excel_filename)
print(f"美國 GDP 資料已儲存至 '{DataFolder + excel_filename}'")
print(gdp_data.head())

# 顯示數據
plt.rcParams['font.family'] = 'Microsoft JhengHei' # 設置中文字體
gdp_data['GDP'].plot() # 畫出圖形
plt.xlabel("Date") # x 軸的標題
plt.ylabel("USA GDP") # y 軸的標題
plt.title("美國 GDP") # 圖標題
plt.show()


# 消費者物價指數及其年增率 網址
url = 'https://ws.dgbas.gov.tw/001/Upload/463/relfile/10315/2414/cpispl.xls'

# pip install xlrd
# 直接從 URL 讀取 excel 文件
TW_cpi = pd.read_excel(url, header = 2) # 指定第三行（索引為2）作為欄位名稱
print(TW_cpi.columns) # 檢視所有欄位

TW_cpi = TW_cpi.drop(columns = ['累計平均']) # 移除該欄位
TW_cpi = TW_cpi[:-4] # 移除最後四筆資料
print(TW_cpi.head())

# 轉換為長格式。將指定列變成行，並且通常是將多個列的數據合併成少數幾列
TW_cpi = TW_cpi.melt(id_vars = '民國年', var_name = '月份', 
                     value_name = 'CPI')

# regex 參數的預設值是 True，會將要替換的字串視為正則表達式處理。
TW_cpi['月份'] = TW_cpi['月份'].str.replace('月', '', regex = False) # 轉換月份

# print(TW_cpi[TW_cpi.isna().any(axis = 1)]) # 顯示缺失值資料
TW_cpi['西元年'] = TW_cpi['民國年'] + 1911
TW_cpi = TW_cpi.drop(columns = ['民國年']) # 移除該欄位
TW_cpi['DATE'] = TW_cpi['西元年'].astype(str) + '/' + TW_cpi['月份'] + '/1  00:00:00' # 合併兩時間為新欄位
TW_cpi['DATE'] = pd.to_datetime(TW_cpi['DATE'], format = '%Y/%m/%d %H:%M:%S', 
                                 errors = 'coerce') # 將 'date_str' 欄位轉換為時間格式
TW_cpi = TW_cpi.drop(columns = ['西元年', '月份']) # 移除該欄位
TW_cpi = TW_cpi.set_index(['DATE']) # 設定索引
TW_cpi = TW_cpi.sort_index()
TW_cpi = TW_cpi.loc[Data_Time_Start : Data_Time_End]
TW_cpi['TW_CPI_Rate'] = TW_cpi['CPI'].pct_change() # 算出增長率
print(TW_cpi.head())

excel_filename = 'TW_CPI.xlsx'
TW_cpi.to_excel(DataFolder + excel_filename)
print(f"台灣 CPI 資料已存儲至 '{DataFolder + excel_filename}'")

# 顯示數據
plt.rcParams['font.family'] = 'Microsoft JhengHei' # 設置中文字體
TW_cpi['CPI'].plot() # 畫出圖形
plt.xlabel("Date") # x 軸的標題
plt.ylabel("TW CPI") # y 軸的標題
plt.title("台灣 CPI") # 圖標題
plt.show()


# 台灣基準利率 網址
url = 'https://www.cbc.gov.tw/tw/public/data/a13rate.xls'

# pip install xlrd
# 直接從 URL 讀取 excel 文件
TW_rate = pd.read_excel(url)
print(TW_rate.columns) # 檢視所有欄位

# 儲存原始數據
excel_filename = 'TW_Base_Rate.xlsx'
TW_rate.to_excel(DataFolder + excel_filename)

TW_rate = TW_rate.iloc[3:] # 移除指定欄位上方的列，否則資料會出現空值
TW_rate.columns = TW_rate.iloc[0] # 指定第五行（索引為 4）作欄位名稱
print(TW_rate.columns) # 檢視所有欄位
TW_rate = TW_rate.iloc[1:] # 避免新的欄位被作為資料參與處理，重新劃定資料範圍

# 原先欄位 '年月' (TW_rate['　　　　']) 下的數字，預設格式為字串，需要先轉成整數才能計算
TW_rate['西元年'] = (TW_rate['　　　　'].astype(int) + 191100).astype(str)
TW_rate['DATE'] = TW_rate['西元年'].str[:4] + '/' + TW_rate['西元年'].str[4:] \
    + '/1  00:00:00'
TW_rate['DATE'] = pd.to_datetime(TW_rate['DATE'], format = '%Y/%m/%d %H:%M:%S', 
                                 errors = 'coerce')
TW_rate = TW_rate.set_index(['DATE']) # 設定索引
TW_rate['TW_Rate'] = TW_rate[['機動']][24:] # 若僅留一層 []，資料型態由 DataFrame 轉為 Series
TW_rate = TW_rate[['TW_Rate']]
TW_rate = TW_rate.sort_index()  
TW_rate = TW_rate.loc[Data_Time_Start : Data_Time_End]
print(TW_rate.head())

excel_filename = 'TW_Rate.xlsx'
TW_rate.to_excel(DataFolder + excel_filename)
print(f"台灣 公告機動利率 資料已儲存至 '{DataFolder + excel_filename}'")

# 顯示數據
plt.rcParams['font.family'] = 'Microsoft JhengHei' # 設置中文字體
TW_rate['TW_Rate'].plot() # 畫出圖形
plt.xlabel("Date") # x 軸的標題
plt.ylabel("TW Base Rate") # y 軸的標題
plt.title("台灣 機動利率") # 圖標題
plt.show()

