import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager
import json
from datetime import datetime

# 设置中文字体支持
try:
    # 尝试使用系统中已有的中文字体
    font_list = [f.name for f in font_manager.fontManager.ttflist]
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
    for font in chinese_fonts:
        if font in font_list:
            plt.rcParams['font.sans-serif'] = [font]
            break
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    print("警告: 中文字体设置失败，图表中的中文可能无法正常显示")

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Fractal:
    """分型类"""
    def __init__(self, k_index: int, fractal_type: str, price: float, date):
        """
        初始化分型
        :param k_index: 分型在原始K线数据中的索引
        :param fractal_type: 分型类型 ('top' 或 'bottom')
        :param price: 分型价格 (顶分型为高点，底分型为低点)
        :param date: 分型对应日期
        """
        self.k_index = k_index
        self.type = fractal_type
        self.price = price
        self.date = date
    
    def __repr__(self):
        return f"{self.type.capitalize()}Fractal(k_index={self.k_index}, price={self.price}, date={self.date})"
    
    def __eq__(self, other):
        if not isinstance(other, Fractal):
            return False
        return (self.k_index == other.k_index and 
                self.type == other.type and 
                abs(self.price - other.price) < 1e-6)
    
    def to_dict(self):
        """将分型对象转换为字典"""
        return {
            'k_index': self.k_index,
            'type': self.type,
            'price': self.price,
            'date': self.date.strftime('%Y-%m-%d') if hasattr(self.date, 'strftime') else str(self.date)
        }
    
    @classmethod
    def from_dict(cls, data):
        """从字典创建分型对象"""
        return cls(
            k_index=data['k_index'],
            fractal_type=data['type'],
            price=data['price'],
            date=pd.to_datetime(data['date'])
        )

class Bi:
    """笔类"""
    def __init__(self, start_fractal: Fractal, end_fractal: Fractal):
        """
        初始化笔
        :param start_fractal: 笔的起始分型
        :param end_fractal: 笔的结束分型
        """
        self.start = start_fractal
        self.end = end_fractal
        
        # 确定笔的方向
        if start_fractal.type == "bottom" and end_fractal.type == "top":
            self.direction = "up"
        elif start_fractal.type == "top" and end_fractal.type == "bottom":
            self.direction = "down"
        else:
            self.direction = "invalid"
    
    def is_valid(self, min_k_gap: int = 5) -> bool:
        """检查笔是否有效"""
        # 检查分型类型配对
        if self.direction == "invalid":
            return False
        
        # 检查K线间隔
        k_gap = abs(self.end.k_index - self.start.k_index)
        if k_gap < min_k_gap:
            return False
        
        # 检查价格方向
        if self.direction == "up":
            return self.end.price > self.start.price
        else:
            return self.end.price < self.start.price
    
    def __repr__(self):
        if self.direction == "invalid":
            return f"InvalidBi(start={self.start}, end={self.end})"
        return f"Bi({self.direction}, start={self.start.price}, end={self.end.price}, from {self.start.date} to {self.end.date})"
    
    def __eq__(self, other):
        if not isinstance(other, Bi):
            return False
        return (self.start == other.start and 
                self.end == other.end and 
                self.direction == other.direction)
    
    def to_dict(self):
        """将笔对象转换为字典"""
        return {
            'start': self.start.to_dict(),
            'end': self.end.to_dict(),
            'direction': self.direction
        }
    
    @classmethod
    def from_dict(cls, data):
        """从字典创建笔对象"""
        return cls(
            start_fractal=Fractal.from_dict(data['start']),
            end_fractal=Fractal.from_dict(data['end'])
        )

class BiIdentifier:
    """笔识别器"""
    def __init__(self, k_data: pd.DataFrame, min_k_gap: int = 5, verbose: bool = False):
        """
        初始化笔识别器
        :param k_data: 已经处理过包含关系的K线数据
        :param min_k_gap: 最小K线间隔
        :param verbose: 是否输出详细日志
        """
        # 数据预处理
        self.k_data = k_data.sort_values('Date').reset_index(drop=True)
        self.min_k_gap = min_k_gap
        self.verbose = verbose
        
        # 检查必要字段
        required_columns = ['Date', 'High', 'Low']
        for col in required_columns:
            if col not in self.k_data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # 转换日期格式
        self.k_data['Date'] = pd.to_datetime(self.k_data['Date'])
        
        # 确保价格列是数值类型
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in self.k_data.columns:
                self.k_data[col] = pd.to_numeric(self.k_data[col], errors='coerce')
        
        self.all_fractals = []  # 所有分型
        self.valid_fractals = []  # 有效分型
        self.bi_list = []  # 笔序列
    
    def identify_fractals(self, start_index: int = 0) -> List[Fractal]:
        """识别所有分型"""
        fractals = []
        n = len(self.k_data)
        
        # 如果start_index>0，需要确保包含前一根K线用于比较
        start_idx = max(0, start_index - 1)
        
        for i in range(start_idx + 1, n-1):
            high_i = self.k_data.loc[i, 'High']
            low_i = self.k_data.loc[i, 'Low']
            high_prev = self.k_data.loc[i-1, 'High']
            low_prev = self.k_data.loc[i-1, 'Low']
            high_next = self.k_data.loc[i+1, 'High']
            low_next = self.k_data.loc[i+1, 'Low']
            
            # 检查是否为有效数值
            if pd.isna(high_i) or pd.isna(low_i) or pd.isna(high_prev) or pd.isna(low_prev) or pd.isna(high_next) or pd.isna(low_next):
                continue
            
            is_top = high_i >= high_prev and high_i >= high_next
            is_bottom = low_i <= low_prev and low_i <= low_next
            
            # 处理同时满足顶底分型的情况
            if is_top and is_bottom:
                # 使用保守策略：跳过模糊分型
                if self.verbose:
                    logger.info(f"Skip ambiguous fractal at index {i}, date {self.k_data.loc[i, 'Date']}")
                continue
            
            if is_top:
                fractals.append(Fractal(
                    k_index=i,
                    fractal_type='top',
                    price=high_i,
                    date=self.k_data.loc[i, 'Date']
                ))
            elif is_bottom:
                fractals.append(Fractal(
                    k_index=i,
                    fractal_type='bottom',
                    price=low_i,
                    date=self.k_data.loc[i, 'Date']
                ))
        
        # 如果是从中间开始识别，保留之前的分型
        if start_index > 0 and self.all_fractals:
            # 找到最后一个在start_index之前的分型
            last_old_index = 0
            for i, fractal in enumerate(self.all_fractals):
                if fractal.k_index < start_index:
                    last_old_index = i
                else:
                    break
            
            # 保留旧分型，添加新分型
            self.all_fractals = self.all_fractals[:last_old_index+1] + fractals
        else:
            self.all_fractals = fractals
            
        if self.verbose:
            logger.info(f"Identified {len(fractals)} new fractals, total {len(self.all_fractals)} fractals")
        return self.all_fractals
    
    def filter_fractals(self, start_index: int = 0) -> List[Fractal]:
        """过滤无效分型，确保分型交替且间隔足够"""
        if not self.all_fractals:
            self.identify_fractals(start_index)
        
        if len(self.all_fractals) < 2:
            return self.all_fractals
        
        # 如果是从中间开始过滤，保留之前有效的分型
        if start_index > 0 and self.valid_fractals:
            # 找到最后一个在start_index之前的分型
            last_old_index = 0
            for i, fractal in enumerate(self.valid_fractals):
                if fractal.k_index < start_index:
                    last_old_index = i
                else:
                    break
            
            # 保留旧的有效分型
            valid_fractals = self.valid_fractals[:last_old_index+1]
        else:
            valid_fractals = [self.all_fractals[0]]
        
        # 从最后一个有效分型之后开始处理
        start_i = 0
        for i, fractal in enumerate(self.all_fractals):
            if fractal.k_index >= start_index:
                start_i = i
                break
        
        # 处理剩余的分型
        for i in range(start_i, len(self.all_fractals)):
            current = self.all_fractals[i]
            
            if not valid_fractals:
                valid_fractals.append(current)
                continue
                
            last = valid_fractals[-1]
            
            # 检查分型类型是否相同
            if current.type == last.type:
                # 同类分型，保留更极端的一个
                if current.type == 'top':
                    if current.price > last.price:
                        valid_fractals[-1] = current  # 替换为更高的顶
                        if self.verbose:
                            logger.info(f"Replace top fractal at {last.k_index} with higher top at {current.k_index}")
                else:
                    if current.price < last.price:
                        valid_fractals[-1] = current  # 替换为更低的底
                        if self.verbose:
                            logger.info(f"Replace bottom fractal at {last.k_index} with lower bottom at {current.k_index}")
            else:
                # 不同类型分型，检查间隔是否足够
                k_gap = abs(current.k_index - last.k_index)
                if k_gap >= self.min_k_gap:
                    valid_fractals.append(current)
                elif self.verbose:
                    logger.info(f"Skip fractal at {current.k_index} due to insufficient gap ({k_gap} < {self.min_k_gap})")
        
        self.valid_fractals = valid_fractals
        if self.verbose:
            logger.info(f"Filtered to {len(valid_fractals)} valid fractals")
            
            # 验证分型交替性
            for i in range(1, len(valid_fractals)):
                if valid_fractals[i].type == valid_fractals[i-1].type:
                    logger.warning(f"Invalid consecutive {valid_fractals[i].type} fractals at indices {valid_fractals[i-1].k_index} and {valid_fractals[i].k_index}")
        
        return valid_fractals
    
    def generate_bi_sequence(self, start_index: int = 0) -> List[Bi]:
        """生成笔序列，实现动态修正机制"""
        if not self.valid_fractals:
            self.filter_fractals(start_index)
        
        if len(self.valid_fractals) < 2:
            return []
        
        # 如果是从中间开始生成，保留之前的笔
        if start_index > 0 and self.bi_list:
            # 找到最后一个在start_index之前的笔
            last_old_index = 0
            for i, bi in enumerate(self.bi_list):
                if bi.end.k_index < start_index:
                    last_old_index = i
                else:
                    break
            
            # 保留旧的笔
            bi_list = self.bi_list[:last_old_index+1]
            # 设置当前起点为最后一笔的终点
            if bi_list:
                current_start = bi_list[-1].end
            else:
                current_start = self.valid_fractals[0]
        else:
            bi_list = []
            current_start = self.valid_fractals[0]
        
        # 找到当前起点在有效分型中的位置
        start_i = 0
        for i, fractal in enumerate(self.valid_fractals):
            if fractal.k_index >= current_start.k_index:
                start_i = i
                break
        
        i = start_i
        
        while i < len(self.valid_fractals) - 1:
            # 确定当前笔的方向
            if current_start.type == 'bottom':
                target_type = 'top'  # 寻找顶分型作为终点
                direction = 'up'
            else:
                target_type = 'bottom'  # 寻找底分型作为终点
                direction = 'down'
            
            # 寻找下一个有效终点
            found_end = False
            potential_ends = []
            
            for j in range(i + 1, len(self.valid_fractals)):
                candidate = self.valid_fractals[j]
                
                # 检查分型类型是否符合预期
                if candidate.type != target_type:
                    # 遇到相反类型的分型，需要检查是否需要修正前一笔
                    if direction == 'up' and candidate.type == 'top' and candidate.price > current_start.price:
                        # 上升笔过程中遇到更高的顶分型，需要修正
                        if self.verbose:
                            logger.info(f"Found higher top during up BI search: {candidate.price} > {current_start.price}")
                        # 更新起点为这个更高的顶分型，重新开始寻找下降笔
                        current_start = candidate
                        i = j
                        break
                    elif direction == 'down' and candidate.type == 'bottom' and candidate.price < current_start.price:
                        # 下降笔过程中遇到更低的底分型，需要修正
                        if self.verbose:
                            logger.info(f"Found lower bottom during down BI search: {candidate.price} < {current_start.price}")
                        # 更新起点为这个更低的底分型，重新开始寻找上升笔
                        current_start = candidate
                        i = j
                        break
                    continue
                
                # 检查K线间隔
                k_gap = abs(candidate.k_index - current_start.k_index)
                if k_gap < self.min_k_gap:
                    continue
                
                # 检查价格方向
                if direction == 'up' and candidate.price > current_start.price:
                    potential_ends.append((j, candidate))
                elif direction == 'down' and candidate.price < current_start.price:
                    potential_ends.append((j, candidate))
            
            # 如果没有找到有效终点，移动到下一个分型
            if not potential_ends:
                i += 1
                if i < len(self.valid_fractals):
                    current_start = self.valid_fractals[i]
                continue
            
            # 选择第一个有效终点
            end_index, end_fractal = potential_ends[0]
            
            # 创建笔
            bi = Bi(current_start, end_fractal)
            if bi.is_valid(self.min_k_gap):
                bi_list.append(bi)
                if self.verbose:
                    logger.info(f"Created valid BI: {bi}")
                
                # 更新当前起点为当前笔的终点
                current_start = end_fractal
                i = end_index
            
            # 如果已经处理完所有分型，退出循环
            if i >= len(self.valid_fractals) - 1:
                break
        
        # 检查笔序列的连续性
        if self.verbose and len(bi_list) > 1:
            for idx in range(1, len(bi_list)):
                if bi_list[idx-1].end.k_index != bi_list[idx].start.k_index:
                    logger.warning(f"BI sequence discontinuity at index {idx}: {bi_list[idx-1].end.k_index} != {bi_list[idx].start.k_index}")
        
        self.bi_list = bi_list
        return bi_list
    
    def identify_bi(self, start_index: int = 0) -> List[Bi]:
        """完整的笔识别流程"""
        self.identify_fractals(start_index)
        self.filter_fractals(start_index)
        self.generate_bi_sequence(start_index)
        return self.bi_list

def process_csv_file(csv_path: str, min_k_gap: int = 5, verbose: bool = False, 
                    incremental: bool = True, state_dir: str = "state") -> Tuple[List[Bi], pd.DataFrame, BiIdentifier]:
    """
    处理CSV文件并识别笔序列，支持增量处理
    :param csv_path: CSV文件路径
    :param min_k_gap: 最小K线间隔
    :param verbose: 是否输出详细日志
    :param incremental: 是否使用增量处理
    :param state_dir: 状态文件保存目录
    :return: 笔序列, K线数据, 笔识别器
    """
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
    
    # 读取CSV文件
    try:
        k_data = pd.read_csv(csv_path)
        print(f"成功读取CSV文件: {csv_path}")
        print(f"数据形状: {k_data.shape}")
    except Exception as e:
        raise ValueError(f"读取CSV文件失败: {e}")
    
    # 检查必要列
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
    for col in required_columns:
        if col not in k_data.columns:
            raise ValueError(f"CSV文件中缺少必要列: {col}")
    
    # 转换日期格式
    k_data['Date'] = pd.to_datetime(k_data['Date'])
    
    # 确保价格列是数值类型
    for col in ['Open', 'High', 'Low', 'Close']:
        k_data[col] = pd.to_numeric(k_data[col], errors='coerce')
    
    # 删除包含NaN值的行
    k_data = k_data.dropna(subset=['Open', 'High', 'Low', 'Close'])
    
    # 创建状态目录
    os.makedirs(state_dir, exist_ok=True)
    
    # 获取股票代码
    symbol = os.path.splitext(os.path.basename(csv_path))[0]
    state_file = os.path.join(state_dir, f"{symbol}_state.json")
    
    # 确定起始处理位置
    start_index = 0
    if incremental and os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # 获取最后处理的位置
            last_processed_index = state.get('last_processed_index', 0)
            last_processed_date = state.get('last_processed_date')
            
            # 找到最后处理日期在数据中的位置
            if last_processed_date:
                last_date = pd.to_datetime(last_processed_date)
                new_data = k_data[k_data['Date'] > last_date]
                if not new_data.empty:
                    start_index = new_data.index[0] - 3  # 多处理几根K线以确保连续性
                    start_index = max(0, start_index)
                    print(f"增量处理: 从索引 {start_index} 开始处理新数据")
                else:
                    print("没有新数据需要处理")
                    return [], k_data, None
            else:
                # 使用索引方式
                if last_processed_index < len(k_data) - 1:
                    start_index = last_processed_index - 3  # 多处理几根K线以确保连续性
                    start_index = max(0, start_index)
                    print(f"增量处理: 从索引 {start_index} 开始处理新数据")
                else:
                    print("没有新数据需要处理")
                    return [], k_data, None
                    
        except Exception as e:
            print(f"读取状态文件失败: {e}, 将进行全量处理")
            start_index = 0
    
    # 创建笔识别器
    bi_identifier = BiIdentifier(k_data, min_k_gap=min_k_gap, verbose=verbose)
    
    # 识别笔序列
    bi_sequence = bi_identifier.identify_bi(start_index)
    
    # 保存状态
    if incremental:
        try:
            # 获取最后处理的索引和日期
            if not k_data.empty:
                last_index = len(k_data) - 1
                last_date = k_data.iloc[-1]['Date']
                
                state = {
                    'last_processed_index': last_index,
                    'last_processed_date': last_date.strftime('%Y-%m-%d'),
                    'processed_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                with open(state_file, 'w') as f:
                    json.dump(state, f, indent=2)
                
                print(f"状态已保存: 最后处理索引 {last_index}, 日期 {last_date}")
        except Exception as e:
            print(f"保存状态文件失败: {e}")
    
    return bi_sequence, k_data, bi_identifier

def save_bi_sequence_to_csv(bi_sequence: List[Bi], output_path: str):
    """
    将笔序列保存到CSV文件
    :param bi_sequence: 笔序列
    :param output_path: 输出文件路径
    """
    # 准备数据
    data = []
    for i, bi in enumerate(bi_sequence):
        data.append({
            'BI_Index': i + 1,
            'Direction': bi.direction,
            'Start_Date': bi.start.date,
            'Start_Price': bi.start.price,
            'End_Date': bi.end.date,
            'End_Price': bi.end.price,
            'Price_Change': bi.end.price - bi.start.price if bi.direction == 'up' else bi.start.price - bi.end.price,
            'K_Line_Count': abs(bi.end.k_index - bi.start.k_index)
        })
    
    # 创建DataFrame并保存
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"笔序列已保存到: {output_path}")

def save_bi_sequence_to_json(bi_sequence: List[Bi], output_path: str):
    """
    将笔序列保存到JSON文件
    :param bi_sequence: 笔序列
    :param output_path: 输出文件路径
    """
    # 转换为字典列表
    bi_dicts = [bi.to_dict() for bi in bi_sequence]
    
    # 保存到JSON文件
    with open(output_path, 'w') as f:
        json.dump(bi_dicts, f, indent=2)
    
    print(f"笔序列已保存到JSON文件: {output_path}")

def load_bi_sequence_from_json(input_path: str) -> List[Bi]:
    """
    从JSON文件加载笔序列
    :param input_path: 输入文件路径
    :return: 笔序列
    """
    if not os.path.exists(input_path):
        return []
    
    try:
        with open(input_path, 'r') as f:
            bi_dicts = json.load(f)
        
        return [Bi.from_dict(bi_dict) for bi_dict in bi_dicts]
    except Exception as e:
        print(f"加载笔序列失败: {e}")
        return []

# 示例用法
def main():
    input_dir = "processed_data"
    output_dir = "sequence_data"
    state_dir = "state"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)
    
    # 获取股票代码列表
    # 这里需要实现get_stock_name函数或直接读取文件列表
    # 假设我们从processed_data目录读取所有CSV文件
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        symbol = os.path.splitext(csv_file)[0]
        input_file = os.path.join(input_dir, csv_file)
        output_file = os.path.join(output_dir, f"{symbol}.csv")
        output_json_file = os.path.join(output_dir, f"{symbol}.json")
        
        print(f"处理股票 {symbol} 的数据")
        
        # 检查是否已经有处理结果
        existing_bi_sequence = []
        if os.path.exists(output_json_file):
            existing_bi_sequence = load_bi_sequence_from_json(output_json_file)
            print(f"找到 {len(existing_bi_sequence)} 条已有的笔序列")
        
        # 处理CSV文件并识别笔序列
        try:
            bi_sequence, k_data, bi_identifier = process_csv_file(
                input_file, 
                min_k_gap=3, 
                verbose=True,
                incremental=True,
                state_dir=state_dir
            )
            
            # 输出结果
            if existing_bi_sequence:
                new_bi_count = len(bi_sequence) - len(existing_bi_sequence)
                if new_bi_count > 0:
                    print(f"识别到 {new_bi_count} 条新笔:")
                    for i in range(len(existing_bi_sequence), len(bi_sequence)):
                        print(f"新笔 {i+1}: {bi_sequence[i]}")
                else:
                    print("没有识别到新笔")
            else:
                print(f"识别到 {len(bi_sequence)} 笔:")
                for i, bi in enumerate(bi_sequence):
                    print(f"笔 {i+1}: {bi}")
            
            # 保存笔序列到CSV和JSON
            if bi_sequence:
                save_bi_sequence_to_csv(bi_sequence, output_file)
                save_bi_sequence_to_json(bi_sequence, output_json_file)
            
        except Exception as e:
            print(f"处理过程中发生错误: {e}")
if __name__ == "__main__":
    main()
