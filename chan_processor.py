import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


def get_latest_data(symbol, processed_file_path=None, days=5):
    """
    从sp500_data文件夹中获取指定股票的最新数据
    :param symbol: 股票代码
    :param processed_file_path: 已处理数据的文件路径（用于确定哪些数据是新的）
    :param days: 如果没有已处理数据，获取最近多少天的数据
    :return: 包含最新数据的DataFrame
    """
    # 构建文件路径
    file_path = f"sp500_data/{symbol}.csv"

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"警告: 文件 {file_path} 不存在")
        return pd.DataFrame()

    try:
        # 读取原始数据
        df = pd.read_csv(file_path)

        # 确保数据按日期排序
        df.sort_values('Date', inplace=True)

        # 如果有已处理数据，找出新增的数据
        if processed_file_path and os.path.exists(processed_file_path):
            # 读取已处理数据
            processed_df = pd.read_csv(processed_file_path)

            # 获取已处理数据的最新日期
            if not processed_df.empty:
                last_processed_date = processed_df['Date'].max()

                # 找出原始数据中比已处理数据新的记录
                new_data = df[df['Date'] > last_processed_date].copy()

                if not new_data.empty:
                    print(f"找到 {len(new_data)} 条新数据")
                    return new_data
                else:
                    print("没有找到新数据")
                    return pd.DataFrame()

        # 如果没有已处理数据或没有找到新数据，返回最后days天的数据
        latest_data = df.tail(days).copy()
        latest_data.reset_index(drop=True, inplace=True)

        return latest_data

    except Exception as e:
        print(f"读取 {file_path} 时出错: {str(e)}")
        return pd.DataFrame()


class ChanTheoryProcessor:
    def __init__(self, data_source):
        """
        初始化缠论处理器
        :param data_source: CSV文件路径或DataFrame
        """
        if isinstance(data_source, str):
            self.df = pd.read_csv(data_source)
        elif isinstance(data_source, pd.DataFrame):
            self.df = data_source.copy()
        else:
            raise ValueError("Unsupported data source type")

        # 处理空值 - 删除包含空值的行
        self.df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

        # 确保数据按日期升序排列
        self.df.sort_values('Date', inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # 确保每根K线的最高价和最低价是正确的
        self._ensure_high_low_correct()

        # 添加标记列
        self.df['Merged'] = False
        self.df['OriginalDates'] = self.df['Date'].astype(str)  # 初始化为单个日期

    def _ensure_high_low_correct(self):
        """
        确保每根K线的最高价是Open和Close中的最大值，最低价是Open和Close中的最小值
        """
        # 计算正确的最高价和最低价
        correct_high = self.df[['Open', 'Close']].max(axis=1)
        correct_low = self.df[['Open', 'Close']].min(axis=1)

        # 检查并修正High和Low列
        high_needs_fix = self.df['High'] < correct_high
        low_needs_fix = self.df['Low'] > correct_low

        if high_needs_fix.any():
            print(f"警告: 发现{high_needs_fix.sum()}根K线的最高价低于实际最高价，已自动修正")
            self.df.loc[high_needs_fix, 'High'] = correct_high[high_needs_fix]

        if low_needs_fix.any():
            print(f"警告: 发现{low_needs_fix.sum()}根K线的最低价高于实际最低价，已自动修正")
            self.df.loc[low_needs_fix, 'Low'] = correct_low[low_needs_fix]

    def is_contained(self, k1, k2):
        """
        判断两根K线是否存在包含关系
        """
        return ((k1['High'] >= k2['High'] and k1['Low'] <= k2['Low']) or
                (k1['High'] <= k2['High'] and k1['Low'] >= k2['Low']))

    def determine_trend(self, prev, last):
        """
        根据前两根K线确定趋势方向
        """
        if prev is None:
            return 'up'  # 起始情况默认向上

        if prev['High'] < last['High'] and prev['Low'] < last['Low']:
            return 'up'
        elif prev['High'] > last['High'] and prev['Low'] > last['Low']:
            return 'down'
        else:
            # 无法判断时，根据价格重心判断
            prev_center = (prev['High'] + prev['Low']) / 2
            last_center = (last['High'] + last['Low']) / 2
            return 'up' if last_center > prev_center else 'down'

    def merge_kline(self, k1, k2, direction):
        """
        合并两根K线
        """
        # 确定合并后的高低点
        if direction == 'up':
            new_high = max(k1['High'], k2['High'])
            new_low = max(k1['Low'], k2['Low'])
        else:  # down
            new_high = min(k1['High'], k2['High'])
            new_low = min(k1['Low'], k2['Low'])

        # 确定开盘价和收盘价 - 确保在高低点范围内
        k1_first = pd.to_datetime(k1['Date']) < pd.to_datetime(k2['Date'])

        if k1_first:
            # k1在前，k2在后
            new_open = k1['Open']
            new_close = k2['Close']
        else:
            # k2在前，k1在后
            new_open = k2['Open']
            new_close = k1['Close']

        # 确保开盘价和收盘价在高低点范围内
        new_open = max(new_low, min(new_high, new_open))
        new_close = max(new_low, min(new_high, new_close))

        # 合并原始日期范围
        if '→' in str(k1.get('OriginalDates', '')) or '→' in str(k2.get('OriginalDates', '')):
            # 如果其中一根已经是合并K线，则合并日期范围
            original_dates = f"{k1.get('OriginalDates', k1['Date'])} → {k2.get('OriginalDates', k2['Date'])}"
        else:
            # 都是原始K线
            original_dates = f"{k1['Date']} → {k2['Date']}"

        return {
            'Date': max(k1['Date'], k2['Date']),  # 取最新日期作为主日期
            'Open': new_open,
            'High': new_high,
            'Low': new_low,
            'Close': new_close,
            'Volume': k1.get('Volume', 0) + k2.get('Volume', 0),
            'Merged': True,
            'OriginalDates': original_dates  # 保存合并的原始日期范围
        }

    def process_containment(self):
        """
        主处理函数：执行包含关系处理逻辑
        """
        if len(self.df) == 0:
            return pd.DataFrame()

        # 使用列表存储处理后的K线
        processed = []

        # 转换为字典列表便于处理
        data = self.df.to_dict('records')

        i = 0
        while i < len(data):
            current = data[i]

            # 第一根K线直接加入
            if not processed:
                processed.append(current)
                i += 1
                continue

            last = processed[-1]

            # 检查当前K线与最后一根处理后的K线是否存在包含关系
            if self.is_contained(last, current):
                # 获取参考K线（倒数第二根）
                prev = processed[-2] if len(processed) >= 2 else None

                # 确定趋势方向
                trend = self.determine_trend(prev, last)

                # 合并K线
                merged_kline = self.merge_kline(last, current, trend)

                # 移除最后一根K线（已被合并）
                processed.pop()

                # 将合并后的K线加入处理列表
                processed.append(merged_kline)
                i += 1
            else:
                # 不存在包含关系，直接添加当前K线
                processed.append(current)
                i += 1

        # 转换为DataFrame
        result_df = pd.DataFrame(processed)

        # 确保所有列都存在
        for col in self.df.columns:
            if col not in result_df.columns:
                result_df[col] = np.nan

        return result_df

    def update_with_new_data(self, symbol):
        """
        使用新数据更新已处理的数据
        :param symbol: 股票代码
        :return: 更新后的完整数据
        """
        # 获取已处理数据的文件路径
        processed_file = f"processed_data/{symbol}.csv"

        # 获取最新数据
        new_data = get_latest_data(symbol, processed_file)

        if new_data.empty:
            print(f"没有新数据需要处理: {symbol}")
            return self.df

        # 获取最后几根已处理的K线（用于确定趋势）
        last_processed_count = min(5, len(self.df))
        last_processed = self.df.tail(last_processed_count).copy()

        # 将新数据添加到已处理数据的末尾
        combined_data = pd.concat([last_processed, new_data], ignore_index=True)

        # 创建新的处理器处理组合数据
        combined_processor = ChanTheoryProcessor(combined_data)
        updated_data = combined_processor.process_containment()

        # 移除可能重复的K线（保留最新的）
        updated_data = updated_data.drop_duplicates(subset=['Date'], keep='last')

        # 将更新后的数据与原始数据（除最后几根）合并
        if len(self.df) > last_processed_count:
            result = pd.concat([self.df.iloc[:-last_processed_count], updated_data], ignore_index=True)
        else:
            result = updated_data

        return result


def get_stock_name():
    # 这里根据您的实际情况加载股票代码
    # 例如从CSV文件、数据库或API获取
    try:
        stock_name = pd.read_csv('sp500_tickers.csv')
        for symbol in stock_name["Symbol"]:
            yield symbol
    except:
        # 如果文件不存在，使用一些示例股票代码
        example_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "JNJ", "V"]
        for symbol in example_symbols:
            yield symbol


def process_all_stocks():
    input_dir = "sp500_data"
    output_dir = "processed_data"
    os.makedirs(output_dir, exist_ok=True)

    symbol_generator = get_stock_name()
    for symbol in symbol_generator:
        symbol = str(symbol).strip()

        input_file = os.path.join(input_dir, f"{symbol}.csv")
        output_file = os.path.join(output_dir, f"{symbol}.csv")

        if not os.path.exists(input_file):
            print(f"跳过:{symbol}.csv 不存在")
            continue

        print(f"处理股票: {symbol}")

        try:
            # 检查输出文件是否存在（是否已经处理过）
            if os.path.exists(output_file):
                # 加载已处理的数据
                processed_df = pd.read_csv(output_file)

                # 使用更新功能处理新数据
                processor = ChanTheoryProcessor(processed_df)
                updated_data = processor.update_with_new_data(symbol)

                # 保存更新后的数据
                updated_data.to_csv(output_file, index=False)
                print(f"已更新 {symbol} 的数据")

            else:
                # 第一次处理，进行完整处理
                sample_df = pd.read_csv(input_file)
                processor = ChanTheoryProcessor(sample_df)
                processed_data = processor.process_containment()
                processed_data.to_csv(output_file, index=False)
                print(f"已完成 {symbol} 的完整处理")

        except Exception as e:
            print(f"处理{symbol}时出错: {str(e)}")

    print("数据全部处理完成")


if __name__ == "__main__":
    process_all_stocks()
