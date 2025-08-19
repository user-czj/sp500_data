import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

DATA_DIR = "sp500_data"

# 遍历 CSV 文件
tickers = [f.split(".")[0] for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

for ticker in tickers:
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    start_date = None

    # 检查文件是否存在
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)

            # 文件为空，直接从零开始下载
            if df.empty:
                start_date = None
            else:
                # 尝试识别日期列
                if 'Date' in df.columns:
                    date_col = 'Date'
                elif 'date' in df.columns:
                    date_col = 'date'
                    df.rename(columns={'date':'Date'}, inplace=True)
                else:
                    # 没有列名或列名不标准，默认第一列为日期
                    df.columns = ['Date','Open','High','Low','Close','Volume','Ticker'][:len(df.columns)]
                    date_col = 'Date'

                # 获取最后日期作为增量起点
                last_date = df[date_col].max()
                start_date = (pd.to_datetime(last_date) + timedelta(days=1)).strftime('%Y-%m-%d')

        except Exception as e:
            print(f"{ticker} 读取 CSV 失败: {e}, 将重新下载全部数据")
            start_date = None
    else:
        # 文件不存在，下载全部数据
        start_date = None

    # 下载增量数据
    try:
        stock = yf.Ticker(ticker)
        if start_date:
            new_data = stock.history(start=start_date, auto_adjust=False).reset_index()
        else:
            new_data = stock.history(period="max", auto_adjust=False).reset_index()

        if new_data.empty:
            print(f"{ticker} 无新数据")
            continue

        # 保留所需列
        new_data['Date'] = pd.to_datetime(new_data['Date']).dt.strftime('%Y-%m-%d')
        keep_columns = ['Date','Open','High','Low','Close','Volume']
        new_data = new_data[[col for col in keep_columns if col in new_data.columns]]

        # 追加到 CSV
        if os.path.exists(file_path):
            new_data.to_csv(file_path, mode='a', index=False, header=False)
        else:
            new_data.to_csv(file_path, index=False, header=True)

        print(f"{ticker} 已追加 {len(new_data)} 条数据")

    except Exception as e:
        print(f"{ticker} 下载或追加数据失败: {e}")
