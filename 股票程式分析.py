import yfinance as yf  # 股票數據模組
import datetime as dt  # 日期模組
import matplotlib.pyplot as plt  # 圖表模組
import numpy as np  # 數值處理模組
from sklearn.linear_model import LinearRegression  # 線性回歸模型

# 獲取股價資料

def get_stock_data(stocks, start_date, end_date):
    stock_data = {}  # 設定一個儲存股票數據的空字典
    for stock_symbol in stocks:  # 在stocks中找尋指定的股票代號
        
        # 判斷此股票代號是否合理
        try:
            # 不合理則告訴使用者
            if yf.Ticker(stock_symbol).history(period="1d").empty:
                print(f"找不到資料，請確認輸入的股票代碼是否正確")
                continue
            # 合理則下載此股票的資料
            else:
                stock_data[stock_symbol] = yf.download(stock_symbol, start=start_date, end=end_date)
        # 處理例外情況
        except Exception as e:
            print(f"無法獲取{stock_symbol}的數據：{e}")
            continue
    return stock_data  # 回傳股票數據給呼叫者


# 使用過去14個交易日的數據計算RSI
def calculate_rsi(data, window=14):
    close = data['Close']  # 選取收盤價
    delta = close.diff()  # 相鄰兩天收盤價的變化

    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()# 計算平均正變化
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()#計算平均負變化

    rs = gain / loss  # rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# 利用長期EMA、短期EMA、信號線計算macd
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    close = data['Close']# 從提供的股票數據中選取收盤價數據
     
    # 使用span參數指定移動平均的窗口大小，min_periods參數指定在計算第一個移動平均之前需要的最小觀測數
    # adjust=False確保使用等權重的移動平均
    short_ema = close.ewm(span=short_window, min_periods=1, adjust=False).mean()# 計算短期(預設為12天)指數加權移動平均（EMA）
    long_ema = close.ewm(span=long_window, min_periods=1, adjust=False).mean()# 計算長期(預設為26天)指數加權移動平均（EMA）
    
    macd_line = short_ema - long_ema# 從短期EMA減去長期EMA得到MACD線
    signal_line = macd_line.ewm(span=signal_window, min_periods=1, adjust=False).mean() # 計算MACD線的信號線(期間預設為9天)
    
    macd = macd_line - signal_line# 計算macd
    return macd, signal_line


# 數據分析可視化
def visualize_stock_data(stock_data):
    # 擷取字典中所有的數據對
    for stock_symbol, data in stock_data.items():  
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))  # 設定三個同一頁面，不同圖表的尺寸

        # 圖一收盤價與趨勢線
        ax1.plot(data['Close'], label='Stock Price') # 繪製收盤價與標籤

        ax1.set_ylabel('Price (USD)') # Y軸標籤
        x = np.array(range(len(data))) # X對應於股票數據中的每個交易日
        x = x.reshape(-1, 1)  # 使 x 成為一個列向量，以便後續將其用於線性回歸模型的擬合
        
        model = LinearRegression().fit(x, data['Close'])  # 創建並擬合線性回歸模型
        trend_line = model.predict(x)  # 使用模型預測趨勢線
        ax1.plot(data.index, trend_line, label='Trend Line', linestyle='--', color='red', alpha=0.5)  # 繪製趨勢線與標籤，並設定為半透明
        
        ax1.grid(True)  # 顯示網格
        y = data['Close'].values.reshape(-1, 1)  # 將收盤價轉換為 NumPy 數組，重新排列為一個列向量，用於線性回歸

        ax1.legend(loc='upper left') #全部標籤設於左上角


        # 圖二根據收盤價創建RSI指標
        ax2.plot(data.index, calculate_rsi(data), label='RSI', color='green') #引用calculate_rsi函數計算RSI

        ax2.axhline(70, color='red', linestyle='--', label='Overbought', alpha=0.3) #RSI高於70為過買
        ax2.axhline(30, color='green', linestyle='--', label='Oversold', alpha=0.3) #RSI低於30為過賣

        ax2.set_ylabel('Extend') # Y軸設定為程度0到100

        ax2.legend(loc='upper left') #全部標籤設於左上角


        # 圖三創建MACD指標
        macd, signal_line = calculate_macd(data)
        ax3.plot(data.index, macd, label='MACD Line', color='orange')  # 繪製macd快速線
        ax3.plot(data.index, signal_line, label='Signal Line', color='blue')  # 繪製signal慢速線

        histogram = macd-signal_line   # 快與慢的差值

        # 繪製柱狀圖 histogram > 0 柱狀圖為綠色 < 0 為紅色
        # 分成兩個以解決只有一個標籤的問題
        ax3.bar(data.index, histogram, width=0.7, label='Difference > 0', color=np.where(histogram < 0, 'red', 'green'), alpha=0.2) 
        ax3.bar(data.index, histogram, width=0.7, label='Difference < 0', color=np.where(histogram > 0, 'green', 'red'), alpha=0.2)

        ax3.set_ylabel('MACD') # Y軸設為MACD 
        ax3.grid(True)  # 顯示網格
        ax3.legend(loc='upper left') #全部標籤設於左上角


        # 在圖三繪製建議買賣時間點
        buy_signals = np.where(np.logical_and(macd > signal_line, macd.shift(1) <= signal_line.shift(1)))[0] # MACD現在大於信號線，但前一天小於信號線，判斷為建議買入
        sell_signals = np.where(np.logical_and(macd < signal_line, macd.shift(1) >= signal_line.shift(1)))[0] # MACD現在小於信號線，但前一天大於信號線，判斷為建議賣出
        
        # 標記交叉點
        ax3.scatter(data.index[buy_signals], macd[buy_signals], marker='^', color='green', label='Buy Signal',s=30) #確定買入的對應日期，放上網上的綠色三角形代表建議買入
        ax3.scatter(data.index[sell_signals], macd[sell_signals], marker='v', color='red', label='Sell Signal',s=30) #確定買入的對應日期，放上網下的紅色三角形代表建議賣出
        
        plt.suptitle(f"{stock_symbol}", fontsize=30)  # 設置圖表標題
        plt.xlabel("Date", fontsize=16)  # 設置x軸標籤
        plt.show()  # 顯示圖表


#程式執行與錯誤處理
#確保模塊只有在直接運行時才執行相應的程式碼
#模組被直接運行時，__name__ 被設置為 "__main__"；當一個模組被引入到另一個模組時，__name__ 被設置為模組的名稱
if __name__ == "__main__":
    stocks_input = input("請輸入股票代碼（例如：2330.TW 可用空格分割輸入多個代碼)：").split()  # 輸入股票代號列表
    start_date_input = input("請輸入開始日期（YYYY-MM-DD）：") # 輸入開始日期
    end_date_input = input("請輸入結束日期（YYYY-MM-DD）超過當前時間將自動計為當前時間：") # 輸入結束日期

    # 檢驗使用者日期的合理性
    try:
        #將用戶輸入的日期字符串轉換為 datetime 對象
        start_date = dt.datetime.strptime(start_date_input, "%Y-%m-%d")
        end_date = dt.datetime.strptime(end_date_input, "%Y-%m-%d")

        # 若結束日期早於開始日期，直接終止程式
        if end_date <= start_date:
            print("結束日期不能早於或等於開始日期")
            exit()
    except ValueError:
        # 其他日期錯誤，如亂碼，直接終止程式
        print("日期格式錯誤")
        exit()
    stock_data = get_stock_data(stocks_input, start_date, end_date)  # 獲取歷史股票數據
    visualize_stock_data(stock_data)  # 可視化歷史收盤價數據、RSI、MACD