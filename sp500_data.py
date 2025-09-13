import os
import csv
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from bs4 import BeautifulSoup
import random
import json
import yfinance as yf
from fake_useragent import UserAgent
import signal
import sys
import subprocess  # 新增导入

# 配置参数
DATA_DIR = "sp500_data"  # 数据存储目录
MAX_RETRIES = 5  # 最大重试次数
RETRY_DELAY = 10  # 重试延迟(秒)
API_TIMEOUT = 20  # API超时时间(秒)
STATE_FILE = "progress_state.json"  # 进度状态文件
TICKER_LIST_FILE = os.path.join(DATA_DIR, "sp500_tickers.csv")  # 成分股列表文件
TICKER_LIST_CACHE_DAYS = 7  # 成分股列表缓存天数
REPORT_FILE = os.path.join(DATA_DIR, "update_report.txt")  # 更新报告文件

# 创建UserAgent对象
ua = UserAgent()

# 全局变量，用于跟踪当前处理的股票
current_ticker_index = 0
total_tickers = 0
tickers = []
updated_count = 0
failed_tickers = []  # 记录更新失败的股票
is_first_run = True  # 新增：标记是否为第一次运行


def signal_handler(sig, frame):
    """处理中断信号（Ctrl+C）"""
    print("\n检测到程序中断，正在保存进度...")
    save_progress_state(current_ticker_index, total_tickers, updated_count)
    print(f"进度已保存，下次将从第 {current_ticker_index + 1} 只股票继续")

    # 生成中断报告
    generate_update_report()

    sys.exit(0)


def get_random_headers():
    """生成随机请求头"""
    return {
        'User-Agent': ua.random,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'DNT': '1',  # Do Not Track
        'Upgrade-Insecure-Requests': '1'
    }


def get_sp500_tickers():
    """
    从SlickCharts获取标普500成分股列表
    返回股票代码列表
    """
    # 检查缓存文件是否存在且未过期
    if os.path.exists(TICKER_LIST_FILE):
        file_time = datetime.fromtimestamp(os.path.getmtime(TICKER_LIST_FILE))
        if (datetime.now() - file_time).days < TICKER_LIST_CACHE_DAYS:
            try:
                print("使用缓存的成分股列表")
                # 明确指定列名，避免依赖文件中的标题行
                tickers_df = pd.read_csv(TICKER_LIST_FILE, names=['Symbol'], header=0)
                return tickers_df['Symbol'].tolist()
            except Exception as e:
                print(f"读取缓存文件失败: {str(e)}，重新抓取列表")

    # 如果缓存过期、不存在或读取失败，则从SlickCharts抓取
    url = "https://www.slickcharts.com/sp500"
    try:
        print("从SlickCharts获取标普500成分股列表...")

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(url, headers=get_random_headers(), timeout=API_TIMEOUT)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')
                table = soup.find('table', {'class': 'table table-hover table-borderless table-sm'})

                if not table:
                    raise ValueError("成分股表格未找到")

                tickers = []
                for row in table.find_all('tr')[1:]:  # 跳过表头
                    cells = row.find_all('td')
                    if len(cells) > 2:
                        # 股票代码在第三列
                        ticker = cells[2].get_text().strip()
                        if '.' in ticker:
                            ticker = ticker.replace('.', '-')  # 处理特殊字符
                        tickers.append(ticker)

                if len(tickers) < 100:
                    raise ValueError(f"获取的成分股数量不足: {len(tickers)}")

                print(f"成功获取 {len(tickers)} 只标普500成分股")

                # 保存成分股列表到文件
                os.makedirs(DATA_DIR, exist_ok=True)
                with open(TICKER_LIST_FILE, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Symbol'])
                    writer.writerows([[t] for t in tickers])

                return tickers
            except (requests.exceptions.RequestException, ValueError) as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"获取成分股列表失败，尝试重试 ({attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(RETRY_DELAY)
                    continue
                raise

    except Exception as e:
        print(f"获取成分股列表失败: {str(e)}")

        # 尝试使用本地缓存（即使过期）
        if os.path.exists(TICKER_LIST_FILE):
            try:
                print("使用本地缓存的成分股列表")
                # 明确指定列名，避免依赖文件中的标题行
                tickers_df = pd.read_csv(TICKER_LIST_FILE, names=['Symbol'], header=0)
                return tickers_df['Symbol'].tolist()
            except Exception as e:
                print(f"读取缓存文件失败: {str(e)}")

        # 使用内置的备选列表
        print("使用内置备选成分股列表")
        return ['AAPL']


def download_stock_data(ticker, start_date=None):
    """
    下载单只股票历史数据 - 使用yfinance库
    返回包含日期、开盘价、最高价、最低价、收盘价的DataFrame
    """
    for attempt in range(MAX_RETRIES):
        try:
            # 创建股票对象
            stock = yf.Ticker(ticker)

            # 获取历史数据
            if start_date:
                # 增量更新：只获取从开始日期至今的数据
                hist = stock.history(start=start_date, auto_adjust=False)
            else:
                # 首次下载：获取全部历史数据
                hist = stock.history(period="max", auto_adjust=False)

            # 重置索引并重命名列
            hist = hist.reset_index()
            hist = hist.rename(columns={
                'Date': 'Date',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })

            # 转换日期格式
            if not hist.empty:
                hist['Date'] = pd.to_datetime(hist['Date']).dt.strftime('%Y-%m-%d')

            # 添加股票代码列
            hist['Ticker'] = ticker

            # 选择需要的列
            keep_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']
            hist = hist[[col for col in keep_columns if col in hist.columns]]

            return hist

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"下载 {ticker} 数据失败，尝试重试 ({attempt + 1}/{MAX_RETRIES})...")
                time.sleep(RETRY_DELAY)
                continue
            print(f"下载 {ticker} 数据失败: {str(e)}")
            return pd.DataFrame()

    return pd.DataFrame()


def save_progress_state(last_processed, total_stocks, updated_count):
    """保存进度状态到文件"""
    state = {
        'last_processed': last_processed,
        'total_stocks': total_stocks,
        'updated_count': updated_count,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'tickers': tickers  # 保存当前成分股列表
    }
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)
    print(f"进度已保存: 最后处理索引={last_processed}, 总股票数={total_stocks}, 更新数={updated_count}")


def load_progress_state():
    """从文件加载进度状态"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                # 返回状态和成分股列表
                return (
                    state.get('last_processed', 0),
                    state.get('total_stocks', 0),
                    state.get('updated_count', 0),
                    state.get('tickers', [])
                )
        except Exception as e:
            print(f"加载进度状态失败: {str(e)}")
            return 0, 0, 0, []
    return 0, 0, 0, []


def is_market_open():
    """检查市场是否开放（简单版）"""
    # 美国东部时间（纽约）的开放时间
    now_utc = datetime.now(timezone.utc)
    now_est = now_utc - timedelta(hours=5)  # UTC-5 为标准时间，UTC-4 为夏令时

    # 检查是否为周末
    if now_est.weekday() >= 5:  # 5=周六, 6=周日
        return False

    # 检查时间（9:30-16:00 美国东部时间）
    market_open = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_est.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= now_est <= market_close


def get_expected_latest_date():
    """获取预期的最近交易日期"""
    today = datetime.today().date()

    # 如果市场开放，预期最新日期是今天
    if is_market_open():
        return today.strftime('%Y-%m-%d')

    # 如果市场关闭，则尝试获取上一个交易日
    # 简单逻辑：如果是周一，则上一个交易日是周五；如果是周日，则上一个交易日是周五；其他情况是前一天
    weekday = today.weekday()
    if weekday == 0:  # 周一
        last_trading_day = today - timedelta(days=3)
    elif weekday == 6:  # 周日
        last_trading_day = today - timedelta(days=2)
    else:
        last_trading_day = today - timedelta(days=1)

    return last_trading_day.strftime('%Y-%m-%d')


def generate_update_report():
    """生成更新报告，列出未成功更新的股票"""
    global failed_tickers

    # 获取预期的最近交易日期
    expected_latest_date = get_expected_latest_date()
    print(f"预期的最近交易日期: {expected_latest_date}")

    # 检查每只股票的数据文件
    outdated_tickers = []
    missing_tickers = []

    for ticker in tickers:
        file_path = os.path.join(DATA_DIR, f"{ticker}.csv")

        # 检查文件是否存在
        if not os.path.exists(file_path):
            missing_tickers.append(ticker)
            continue

        try:
            # 读取数据文件
            df = pd.read_csv(file_path)
            if df.empty:
                outdated_tickers.append(ticker)
                continue

            # 获取最新日期
            latest_date = df['Date'].max()

            # 检查是否达到预期日期
            if latest_date < expected_latest_date:
                outdated_tickers.append(ticker)

        except Exception as e:
            print(f"检查 {ticker} 数据时出错: {str(e)}")
            outdated_tickers.append(ticker)

    # 生成报告
    report_content = f"标普500数据更新报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report_content += f"预期的最近交易日期: {expected_latest_date}\n"
    report_content += f"总股票数量: {len(tickers)}\n"
    report_content += f"成功更新股票数量: {updated_count}\n"
    report_content += f"处理失败的股票数量: {len(failed_tickers)}\n"
    report_content += f"未达到最新日期的股票数量: {len(outdated_tickers)}\n"
    report_content += f"数据文件缺失的股票数量: {len(missing_tickers)}\n\n"

    if failed_tickers:
        report_content += "=== 处理失败的股票 ===\n"
        report_content += ", ".join(failed_tickers) + "\n\n"

    if outdated_tickers:
        report_content += "=== 未达到最新日期的股票 ===\n"
        report_content += ", ".join(outdated_tickers) + "\n\n"

    if missing_tickers:
        report_content += "=== 数据文件缺失的股票 ===\n"
        report_content += ", ".join(missing_tickers) + "\n"

    # 保存报告到文件
    with open(REPORT_FILE, 'w') as f:
        f.write(report_content)

    # 打印报告摘要
    print("\n" + "=" * 50)
    print("数据更新报告")
    print("=" * 50)
    print(report_content)

    return report_content


def run_chan_processor():
    """运行缠论处理器处理包含关系"""
    print("开始运行缠论处理器处理包含关系...")

    try:
        # 检查是否为第一次运行
        global is_first_run
        processed_dir = "processed_data"

        if is_first_run and not os.path.exists(processed_dir):
            # 第一次运行，需要全量处理
            print("第一次运行，执行全量处理...")
            result = subprocess.run([sys.executable, "chan_processor.py"],
                                    capture_output=True, text=True, timeout=3600)  # 1小时超时
        else:
            # 后续运行，使用增量处理
            print("后续运行，执行增量处理...")
            # 这里可以添加增量处理的逻辑，或者直接运行chan_processor.py
            # 因为chan_processor.py内部已经实现了增量处理的逻辑
            result = subprocess.run([sys.executable, "chan_processor.py"],
                                    capture_output=True, text=True, timeout=1800)  # 30分钟超时

        print("缠论处理器输出:")
        print(result.stdout)
        if result.stderr:
            print("缠论处理器错误:")
            print(result.stderr)

        if result.returncode == 0:
            print("缠论处理器运行成功")
        else:
            print(f"缠论处理器运行失败，返回码: {result.returncode}")

    except subprocess.TimeoutExpired:
        print("缠论处理器运行超时")
    except Exception as e:
        print(f"运行缠论处理器时发生错误: {str(e)}")


def update_all_stocks():
    """更新所有标普500股票数据（支持分多天运行）"""
    global current_ticker_index, total_tickers, tickers, updated_count, failed_tickers, is_first_run

    # 重置失败列表
    failed_tickers = []

    # 加载进度状态
    last_processed, total_stocks, prev_updated_count, prev_tickers = load_progress_state()

    # 获取成分股列表（使用缓存）
    current_tickers = get_sp500_tickers()
    total_current_tickers = len(current_tickers)

    # 检查是否需要重新开始
    if total_stocks != total_current_tickers or prev_tickers != current_tickers:
        print(f"成分股列表已变化，重置进度")
        last_processed = 0
        prev_updated_count = 0
        is_first_run = True  # 成分股列表变化，视为第一次运行
    else:
        print(f"恢复之前的进度: 最后处理索引={last_processed}, 总股票数={total_stocks}, 更新数={prev_updated_count}")
        is_first_run = (last_processed == 0)  # 如果从第0只开始，视为第一次运行

    # 更新全局变量
    current_ticker_index = last_processed
    total_tickers = total_current_tickers
    tickers = current_tickers
    updated_count = prev_updated_count

    print(f"开始处理 {total_tickers} 只标普500成分股 (从 #{current_ticker_index} 继续)...")

    # 创建数据目录
    os.makedirs(DATA_DIR, exist_ok=True)

    # 创建日志文件
    log_file = os.path.join(DATA_DIR, "update_log.csv")
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Ticker', 'LastUpdate', 'Status', 'Records'])

    # 处理股票（从上次停止处开始）
    for i in range(current_ticker_index, total_tickers):
        current_ticker_index = i  # 更新当前索引
        ticker = tickers[i]
        try:
            file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
            start_date = None

            # 检查已有数据
            if os.path.exists(file_path):
                try:
                    existing = pd.read_csv(file_path)
                    if not existing.empty:
                        last_date = existing['Date'].max()
                        start_date = (pd.to_datetime(last_date) + timedelta(days=1)).strftime('%Y-%m-%d')
                        print(f"{ticker}: 已有数据至 {last_date}, 增量更新从 {start_date} 开始")
                except Exception as e:
                    print(f"读取 {ticker} 现有数据文件失败: {str(e)}，重新下载完整数据")
                    start_date = None

            # 下载新数据
            new_data = download_stock_data(ticker, start_date)

            if not new_data.empty:
                # 合并数据
                if start_date and os.path.exists(file_path):
                    try:
                        existing = pd.read_csv(file_path)
                        # 确保没有重复数据
                        last_date = existing['Date'].max()
                        new_data = new_data[new_data['Date'] > last_date]

                        if not new_data.empty:
                            updated_data = pd.concat([existing, new_data])
                            status = "更新成功"
                            records = len(new_data)
                            updated_count += 1
                        else:
                            updated_data = existing
                            status = "无新数据"
                            records = 0
                    except:
                        # 如果合并失败，直接使用新数据
                        updated_data = new_data
                        status = "完整下载成功"
                        records = len(new_data)
                        updated_count += 1
                else:
                    updated_data = new_data
                    status = "完整下载成功"
                    records = len(new_data)
                    updated_count += 1

                # 保存数据
                updated_data.to_csv(file_path, index=False)
            else:
                status = "无新数据"
                records = 0

            print(f"({i + 1}/{total_tickers}) {ticker}: {status} ({records}条记录)")

            # 记录日志
            with open(log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([ticker, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), status, records])

            # 更新进度状态（处理完一只股票后立即保存）
            save_progress_state(i + 1, total_tickers, updated_count)

            # 随机请求间隔避免被封
            time.sleep(random.uniform(1.0, 3.0))

        except Exception as e:
            print(f"处理 {ticker} 时出错: {str(e)}")
            failed_tickers.append(ticker)
            with open(log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([ticker, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), f"错误: {str(e)}", 0])

            # 即使出错，也保存进度（跳过当前股票）
            save_progress_state(i + 1, total_tickers, updated_count)

    # 检查是否完成所有股票
    completion_status = "部分完成" if current_ticker_index < total_tickers - 1 else "全部完成"
    print(
        f"标普500数据更新{completion_status}! 已处理 {current_ticker_index + 1}/{total_tickers} 只股票, 成功更新 {updated_count} 只")

    # 如果完成所有股票，重置进度状态
    if current_ticker_index >= total_tickers - 1:
        print("所有股票处理完成，重置进度状态")
        save_progress_state(0, total_tickers, 0)

    # 生成更新报告
    generate_update_report()


def main():
    """主函数"""
    # 注册中断信号处理
    signal.signal(signal.SIGINT, signal_handler)

    print("=" * 50)
    print(f"标普500数据采集系统启动 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # 自动更新数据
    update_all_stocks()

    # 数据更新完成后运行缠论处理器
    run_chan_processor()

    print("=" * 50)
    print("程序执行完毕")
    print("=" * 50)


if __name__ == "__main__":
    # 安装必要的库
    try:
        import fake_useragent
    except ImportError:
        print("安装必要的依赖库...")
        os.system("pip install fake_useragent pandas requests beautifulsoup4 yfinance")

    try:
        main()
    except Exception as e:
        print(f"程序发生错误: {str(e)}")
        # 在异常退出前保存进度
        save_progress_state(current_ticker_index, total_tickers, updated_count)
        print(f"进度已保存，下次将从第 {current_ticker_index + 1} 只股票继续")

        # 生成更新报告
        generate_update_report()
