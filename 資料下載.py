import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

Data_Time_Start = '2019-01-01'
Data_Time_End = '2023-12-31'
Data_Time_TW_Start = str(int(Data_Time_Start[0 : 4]) - 1911) + '-01-01'
Data_Time_TW_End = str(int(Data_Time_End[0 : 4]) - 1911) + '-12-31'


Currency_symbol = 'TWD%3DX' # 輸入股票代號下載匯率資料
Currency_data = yf.download(Currency_symbol, 
                            start = Data_Time_Start, end = Data_Time_End) # 獲取特定日期範圍的匯率資料

excel_filename = f'{Currency_symbol}_Currency_Data.xlsx' # 將匯率資料存儲為 Excel 檔案，以匯率代號作為檔案名稱
Currency_data.to_excel(excel_filename)
print(f"匯率資料已存儲為 '{excel_filename}'")
print(Currency_data.head())

# 顯示數據
Currency_data['Close'].plot() # 畫出圖形
plt.xlabel("Date") # x 軸的標籤
plt.ylabel("Closing Price") # y 軸的標籤
plt.title("USD -> TWD") # 圖標題
plt.show()


#下載美元指數與其成長率
dxy_data = yf.download("DX-Y.NYB", start=Data_Time_Start, end=Data_Time_End)
dxy_data['Growth Rate'] = dxy_data['Close'].pct_change() # 計算每日的成長率（百分比）
print(dxy_data[['Close', 'Growth Rate']].head())# 輸出結果
dxy_data = dxy_data[['Close', 'Growth Rate']]

excel_filename = 'dxy_data.xlsx'
dxy_data.to_excel(excel_filename)
print("美元指數與成長率已保存為 'dxy_data.xlsx'")

# 顯示數據
dxy_data['Close'].plot() # 畫出圖形
plt.xlabel("Date") # x 軸的標籤
plt.ylabel("Close'") # y 軸的標籤
plt.title("dxy_data") # 圖標題
plt.show()


# 下載黃金價格數據
gold_data = yf.download("GC=F", start=Data_Time_Start, end=Data_Time_End)
gold_data.to_excel('gold_data.xlsx')
gold_data['Growth Rate'] = gold_data['Close'].pct_change() # 計算每日的成長率（百分比）
print(gold_data[['Close', 'Growth Rate']].head())# 輸出結果

excel_filename = 'gold_data.xlsx'
gold_data.to_excel(excel_filename)
print("黃金價格已保存為 'gold_data.xlsx'")

# 顯示數據
gold_data['Close'].plot() # 畫出圖形
plt.xlabel("Date") # x 軸的標籤
plt.ylabel("Close'") # y 軸的標籤
plt.title("gold_data") # 圖標題
plt.show()


import pandas_datareader.data as WebData

# pip install pandas_datareader
# 下載聯邦基金利率數據
fed_funds_rate = WebData.DataReader('FEDFUNDS', 'fred', 
                                    start = Data_Time_Start, 
                                    end = Data_Time_End)

excel_filename = 'Fed_Funds_Rate.xlsx'
fed_funds_rate.to_excel(excel_filename)
print(f"匯率資料已存儲為 '{excel_filename}'")
print(fed_funds_rate.head())

# 顯示數據
fed_funds_rate['FEDFUNDS'].plot() # 畫出圖形
plt.xlabel("Date") # x 軸的標籤
plt.ylabel("FEDFUNDS") # y 軸的標籤
plt.title("Fed Funds Rate") # 圖標題
plt.show()


# 下載美國 CPI 數據
cpi_data = WebData.get_data_fred('CPIAUCNS',
                                       start = Data_Time_Start, 
                                       end = Data_Time_End)
cpi_data['USA_CPI_Rate'] = cpi_data['CPIAUCNS'].pct_change() # 算出增長率

excel_filename = 'USA_CPI_Data.xlsx'
cpi_data.to_excel(excel_filename)
print(f"美國 cpi 資料已存儲為 '{excel_filename}'")
print(cpi_data.head())

# 顯示數據
plt.rcParams['font.family'] = 'Microsoft JhengHei' # 設置中文字體
cpi_data['CPIAUCNS'].plot() # 畫出圖形
plt.xlabel("Date") # x 軸的標籤
plt.ylabel("CPIAUCNS") # y 軸的標籤
plt.title("美國 CPI") # 圖標題
plt.show()


# 下載美國失業率數據
unemployment_rate = WebData.get_data_fred('UNRATE',
                                          start = Data_Time_Start, 
                                          end = Data_Time_End)

excel_filename = 'USA_Unemployment_Rate.xlsx'
unemployment_rate.to_excel(excel_filename)
print(f"美國失業率資料已存儲為 '{excel_filename}'")
print(unemployment_rate.head())

# 顯示數據
plt.rcParams['font.family'] = 'Microsoft JhengHei' # 設置中文字體
unemployment_rate['UNRATE'].plot() # 畫出圖形
plt.xlabel("Date") # x 軸的標籤
plt.ylabel("UNRATE") # y 軸的標籤
plt.title("美國失業率") # 圖標題
plt.show()


# 獲取 GDP 資料
gdp_data = WebData.get_data_fred('GDP',
                                 start = Data_Time_Start, 
                                 end = Data_Time_End)
gdp_data['USA_GDP_Rate'] = gdp_data['GDP'].pct_change() # 算出增長率
excel_filename = 'USA_GDP.xlsx'
gdp_data.to_excel(excel_filename)
print(f"美國失業率資料已存儲為 '{excel_filename}'")
print(gdp_data.head())

# 顯示數據
plt.rcParams['font.family'] = 'Microsoft JhengHei' # 設置中文字體
gdp_data['GDP'].plot() # 畫出圖形
plt.xlabel("Date") # x 軸的標籤
plt.ylabel("GDP") # y 軸的標籤
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
TW_cpi.to_excel(excel_filename)
print(f"台灣 CPI 資料已存儲為 '{excel_filename}'")

# 顯示數據
plt.rcParams['font.family'] = 'Microsoft JhengHei' # 設置中文字體
TW_cpi['CPI'].plot() # 畫出圖形
plt.xlabel("Date") # x 軸的標籤
plt.ylabel("CPI") # y 軸的標籤
plt.title("台灣 CPI") # 圖標題
plt.show()


# 台灣基準利率 網址
url = 'https://www.cbc.gov.tw/tw/public/data/a13rate.xls'

# pip install xlrd
# 直接從 URL 讀取 excel 文件
TW_rate = pd.read_excel(url, header = 4) # 指定第五行（索引為4）作欄位名稱
print(TW_rate.columns) # 檢視所有欄位
print(len(TW_rate))

TW_rate['西元年'] = (TW_rate['　　　　'] + 191100).astype(str)
TW_rate['DATE'] = TW_rate['西元年'].str[:4] + '/' + TW_rate['西元年'].str[4:] + '/1  00:00:00'
TW_rate['DATE'] = pd.to_datetime(TW_rate['DATE'], format = '%Y/%m/%d %H:%M:%S', 
                                 errors = 'coerce')
TW_rate = TW_rate.set_index(['DATE']) # 設定索引
#print(type(TW_rate))
TW_rate['TW_Rate'] = TW_rate[['機動']][24:] # 若僅留一層 []，資料型態由 DataFrame 轉為 Series
TW_rate = TW_rate[['TW_Rate']]
TW_rate = TW_rate.sort_index()  
TW_rate = TW_rate.loc[Data_Time_Start : Data_Time_End]
#print(type(TW_rate))
print(TW_rate.head())

excel_filename = 'TW_Rate.xlsx'
TW_rate.to_excel(excel_filename)
print(f"台灣 公告機動利率 資料已存儲為 '{excel_filename}'")

# 顯示數據
plt.rcParams['font.family'] = 'Microsoft JhengHei' # 設置中文字體
TW_rate.plot() # 畫出圖形
plt.xlabel("Date") # x 軸的標籤
plt.ylabel("Rate") # y 軸的標籤
plt.title("台灣 機動利率") # 圖標題
plt.show()

