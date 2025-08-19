import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

DATA_DIR = "sp500_data"

# 遍历所有 CSV
tickers = [f.split(".")[0] for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

for ticker in tickers:
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    start_date = None

    # 获取已有数据的最新日期
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if not df.empty:
            last_date = df['Date'].max()
            start_date = (pd.to_datetime(last_date) + timedelta(days=1)).strftime('%Y-%m-%d')

    # 下载增量数据
    stock = yf.Ticker(ticker)
    new_data = stock.history(start=start_date, auto_adjust=False).reset_index()
    if not new_data.empty:
        new_data['Date'] = pd.to_datetime(new_data['Date']).dt.strftime('%Y-%m-%d')
        # 保留所需列
        new_data = new_data[['Date','Open','High','Low','Close','Volume']]
        # 追加到原 CSV
        new_data.to_csv(file_path, mode='a', index=False, header=False)
        print(f"{ticker} 已追加 {len(new_data)} 条数据")
    else:
        print(f"{ticker} 无新数据")
