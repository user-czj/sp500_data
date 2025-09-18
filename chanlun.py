# -*- coding: utf-8 -*-
"""
缠论分析批量处理版本
保持原始缠论分析逻辑完全不变，包含完整的三类买点(3买点)逻辑
适配CSV格式: Date,Open,High,Low,Close,Volume,Ticker
"""

import sys
import pandas as pd
import numpy as np
from datetime import date, datetime
import os
import time
import glob

# 设置递归深度
sys.setrecursionlimit(10000)


class ChanAnalysis:
    def __init__(self, data_dir, output_base_dir, debug=1, check_data_freshness=False):
        self.data_dir = data_dir
        self.output_base_dir = output_base_dir
        self.debug = debug
        self.check_data_freshness = check_data_freshness

        # 创建输出目录
        for dir_name in ['buy1', 'buy2', 'buy3', 'buy23', 'sell1', 'sell2', 'sell3', 'seg_buy', 'seg_sell']:
            os.makedirs(os.path.join(output_base_dir, dir_name), exist_ok=True)

    def process_all_files(self):
        """处理所有CSV文件"""
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        print(f"找到 {len(csv_files)} 个CSV文件")

        for i, csv_file in enumerate(csv_files):
            try:
                print(f"处理文件 {i + 1}/{len(csv_files)}: {os.path.basename(csv_file)}")
                self.process_single_file(csv_file)
            except Exception as e:
                print(f"处理文件 {csv_file} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    def process_single_file(self, csv_file):
        """处理单个CSV文件"""
        # 读取数据
        df = pd.read_csv(csv_file)

        # 转换列名和格式
        df = df.rename(columns={'Date': 'datetime', 'Low': 'low', 'High': 'high'})
        df['datetime'] = pd.to_datetime(df['datetime'])

        # 只有当需要检查数据新鲜度时才检查
        if self.check_data_freshness and df['datetime'].iloc[-1].date() < date.today():
            print(f"文件 {os.path.basename(csv_file)} 不是最新数据，跳过")
            return

        # 执行分析
        self.buy_sell(df, csv_file)

    def buy_sell(self, df, csv_file):
        """主要分析函数 - 保持原始逻辑不变"""
        ticker = os.path.basename(csv_file).split('.')[0]

        # 使用部分数据进行调试
        if self.debug == 0:
            self.debug = 1

        df = df[['low', 'high', 'datetime']][:-self.debug].copy()

        if self.debug >= len(df):
            print('数据量不足，跳过')
            return

        print('processing ' + ticker)

        # 初始处理 - 移除初始不符合条件的行
        i = 0
        while True:
            if (df['low'].iloc[i] <= df['low'].iloc[i + 1]) or (df['high'].iloc[i] <= df['high'].iloc[i + 1]):
                i += 1
            else:
                break

        df = df[i:].reset_index(drop=True)

        # 处理包含关系 - 保持原始逻辑
        while True:
            temp_len = len(df)
            i = 0
            while i <= len(df) - 4:
                if (df.iloc[i + 2, 0] >= df.iloc[i + 1, 0] and df.iloc[i + 2, 1] <= df.iloc[i + 1, 1]) or \
                        (df.iloc[i + 2, 0] <= df.iloc[i + 1, 0] and df.iloc[i + 2, 1] >= df.iloc[i + 1, 1]):
                    if df.iloc[i + 1, 0] > df.iloc[i, 0]:
                        df.iloc[i + 2, 0] = max(df.iloc[i + 1:i + 3, 0])
                        df.iloc[i + 2, 1] = max(df.iloc[i + 1:i + 3, 1])
                        df.drop(df.index[i + 1], inplace=True)
                        continue
                    else:
                        df.iloc[i + 2, 0] = min(df.iloc[i + 1:i + 3, 0])
                        df.iloc[i + 2, 1] = min(df.iloc[i + 1:i + 3, 1])
                        df.drop(df.index[i + 1], inplace=True)
                        continue
                i += 1

            if len(df) == temp_len:
                break

        df = df.reset_index(drop=True)

        # 获取顶分型和底分型 - 保持原始逻辑
        ul = [0]
        for i in range(len(df) - 2):
            if df.iloc[i + 2, 0] < df.iloc[i + 1, 0] and df.iloc[i, 0] < df.iloc[i + 1, 0]:
                ul.append(1)
                continue
            if df.iloc[i + 2, 0] > df.iloc[i + 1, 0] and df.iloc[i, 0] > df.iloc[i + 1, 0]:
                ul.append(-1)
                continue
            else:
                ul.append(0)

        ul.append(0)
        df1 = pd.concat((df[['low', 'high']], pd.DataFrame(ul), df['datetime']), axis=1)
        df1.columns = ['low', 'high', 'od', 'datetime']

        i = 0
        while i < len(df1) - 2 and df1.iloc[i, 2] == 0:
            i += 1

        df1 = df1[i:]

        i = 0
        while i < len(df1) - 2 and (sum(abs(df1.iloc[i + 1:i + 4, 2])) > 0 or df1.iloc[i, 2] == 0):
            i += 1

        df1 = df1[i:]

        if len(df1) <= 60:
            print('数据量不足，无法分析')
            return

        # 识别有效的分型点 - 保持原始逻辑
        od_list = [0]
        self.judge(0, 0, 1, df1, od_list)

        # 生成线段 - 保持原始逻辑
        start = 0
        while start < len(od_list) - 5:
            if self.check_init_seg(od_list[start:start + 4], df1):
                break
            else:
                start += 1

        if start >= len(od_list) - 5:
            print('无法找到有效的初始线段')
            return

        lines = []
        i = start
        end = False

        while i <= len(od_list) - 4:
            se = self.Seg(od_list[i:i + 4], df1)
            label = False
            while label == False and i <= len(od_list) - 6:
                i += 2
                label, start_idx = se.grow(od_list[i + 2:i + 4], df1)
                if se.vertex[-1] > od_list[-3]:
                    end = True
                    lines.append(se.lines(df1))
                    break

            if end:
                break

            i = np.where(np.array(od_list) == se.vertex[-1])[0][0]
            lines.append(se.lines(df1))

        # 处理尾部线段 - 保持原始逻辑
        low_list = df1.iloc[se.vertex[-1]:, 0]
        high_list = df1.iloc[se.vertex[-1]:, 1]
        low_extre = low_list.min()
        high_extre = high_list.max()

        if se.finished == True:
            if lines[-1][0][1] < lines[-1][1][1]:
                lines.append([(se.vertex[-1], lines[-1][1][1]), (low_list.idxmin(), low_extre)])
            else:
                lines.append([(se.vertex[-1], lines[-1][1][1]), (high_list.idxmax(), high_extre)])
        else:
            if lines[-1][0][1] < lines[-1][1][1]:
                if low_extre > lines[-1][0][1]:
                    lines[-1] = [(lines[-1][0][0], lines[-1][0][1]), (high_list.idxmax(), high_extre)]
                else:
                    if low_list.idxmin() - se.vertex[-1] >= 10:
                        lines.append([(se.vertex[-1], lines[-1][1][1]), (low_list.idxmin(), low_extre)])
            else:
                if high_extre < lines[-1][0][1]:
                    lines[-1] = [(lines[-1][0][0], lines[-1][0][1]), (low_list.idxmin(), low_extre)]
                else:
                    if high_list.idxmax() - se.vertex[-1] >= 10:
                        lines.append([(se.vertex[-1], lines[-1][1][1]), (high_list.idxmax(), high_extre)])

        # 生成中枢 - 保持原始逻辑
        a, tails = self.get_pivot(lines, df1)
        pro_a = self.process_pivot(a)

        # 生成买卖信号 - 保持原始逻辑
        signal, interval = self.buy_point1(pro_a, tails, df1)
        if signal:
            pro_a[-1].write_out(os.path.join(self.output_base_dir, 'buy1', f'{ticker}_buy1.txt'), tails, df1)

        signal, interval = self.buy_point3_des(pro_a, tails, df1)
        if signal:
            pro_a[-1].write_out(os.path.join(self.output_base_dir, 'buy3', f'{ticker}_buy3.txt'), tails, df1)

        # 添加三类买点(3买点)逻辑
        signal, interval = self.buy_point3(pro_a, tails, df1)
        if signal:
            pro_a[-1].write_out(os.path.join(self.output_base_dir, 'buy3', f'{ticker}_buy3_real.txt'), tails, df1)

        signal, interval = self.buy_point23(pro_a, tails, df1)
        if signal:
            pro_a[-1].write_out(os.path.join(self.output_base_dir, 'buy23', f'{ticker}_buy23.txt'), tails, df1)

        signal, interval = self.buy_point2(pro_a, tails, df1)
        if signal:
            pro_a[-1].write_out(os.path.join(self.output_base_dir, 'buy2', f'{ticker}_buy2.txt'), tails, df1)

        signal, interval = self.sell_point1(pro_a, tails, df1)
        if signal:
            pro_a[-1].write_out(os.path.join(self.output_base_dir, 'sell1', f'{ticker}_sell1.txt'), tails, df1)

        signal, interval = self.sell_point3_ris(pro_a, tails, df1)
        if signal:
            pro_a[-1].write_out(os.path.join(self.output_base_dir, 'sell3', f'{ticker}_sell3.txt'), tails, df1)

        signal, interval = self.sell_point2(pro_a, tails, df1)
        if signal:
            pro_a[-1].write_out(os.path.join(self.output_base_dir, 'sell2', f'{ticker}_sell2.txt'), tails, df1)

        signal, interval = self.seg_buy(lines[-5:], df1)
        if signal:
            self.write_seg(lines, os.path.join(self.output_base_dir, 'seg_buy', f'{ticker}_seg_buy.txt'), True,
                           interval, df1)

        signal, interval = self.seg_sell(lines[-5:], df1)
        if signal:
            self.write_seg(lines, os.path.join(self.output_base_dir, 'seg_sell', f'{ticker}_seg_sell.txt'), False,
                           interval, df1)

    def judge(self, prev_i, cur_i, d, df1, od_list):
        """递归函数，确认分型有效性 - 保持原始逻辑"""
        if cur_i + 4 >= len(df1) - 1:
            return

        if cur_i - prev_i < 4 or df1['od'].iloc[cur_i] != d:
            self.judge(prev_i, cur_i + 1, d, df1, od_list)
        else:
            new_i, label1 = self.exist_new_extreme(cur_i, d, 2, 3, df1)
            if label1:
                self.judge(prev_i, new_i, d, df1, od_list)
            else:
                k = 4
                while cur_i + k < len(df1) - 1 and not self.exist_opposite(cur_i, d, k, df1):
                    new_i, label2 = self.exist_new_extreme(cur_i, d, k, k, df1)
                    if label2:
                        self.judge(prev_i, new_i, d, df1, od_list)
                        return
                    k += 1
                    if cur_i + k >= len(df1) - 1:
                        return

                prev_i = cur_i
                cur_i = cur_i + k
                od_list.append(prev_i)
                self.judge(prev_i, cur_i, -d, df1, od_list)

    def exist_opposite(self, cur_i, d, pos, df1):
        """检查是否存在相反分型 - 保持原始逻辑"""
        return (df1['od'].iloc[cur_i + pos] == -d and
                self.same_d(df1.iloc[cur_i, 0], df1.iloc[cur_i, 1],
                            df1.iloc[cur_i + pos, 0], df1.iloc[cur_i + pos, 1], d))

    def same_d(self, a1, a2, b1, b2, a_sign):
        """检查是否同方向 - 保持原始逻辑"""
        if a_sign == 1:
            return (a1 > b1 and a2 > b2)
        else:
            return (a1 < b1 and a2 < b2)

    def exist_new_extreme(self, cur_i, d, start, end, df1):
        """检查是否存在新高/新低 - 保持原始逻辑"""
        j = start
        while j <= end:
            if self.new_extreme(df1.iloc[cur_i, 0], df1.iloc[cur_i, 1],
                                df1.iloc[cur_i + j, 0], df1.iloc[cur_i + j, 1], d):
                return cur_i + j, True
            j += 1
        return cur_i, False

    def new_extreme(self, a1, a2, b1, b2, a_sign):
        """检查是否创新高/新低 - 保持原始逻辑"""
        if a_sign == 1:
            return b2 >= a2
        else:
            return a1 >= b1

    def check_init_seg(self, start_l, df1):
        """检查初始线段有效性 - 保持原始逻辑"""
        d = -df1.iloc[start_l[0], 2]
        if not ((d == 1 or d == -1) and (len(start_l) == 4)):
            print('initializing seg failed in check_init_seg!')
            return False

        if d == 1:
            if df1.iloc[start_l[1], 1] < df1.iloc[start_l[3], 1] and \
                    df1.iloc[start_l[0], 0] < df1.iloc[start_l[2], 0]:
                return True
            else:
                return False
        else:
            if df1.iloc[start_l[1], 0] > df1.iloc[start_l[3], 0] and \
                    df1.iloc[start_l[0], 1] > df1.iloc[start_l[2], 1]:
                return True
            else:
                return False

    class Seg:
        """线段类 - 保持原始逻辑"""

        def __init__(self, start_l, df1):
            self.start = start_l[0]
            self.df1 = df1  # 保存df1引用

            if df1.iloc[start_l[0], 2] == 0:
                print("error init!")
            self.d = -df1.iloc[start_l[0], 2]

            self.finished = False
            self.vertex = start_l
            self.gap = False

            if self.d == 1:
                self.cur_extreme = df1.iloc[start_l[3], 1]
                self.cur_extreme_pos = start_l[3]
                self.prev_extreme = df1.iloc[start_l[1], 1]
            else:
                self.cur_extreme = df1.iloc[start_l[3], 0]
                self.cur_extreme_pos = start_l[3]
                self.prev_extreme = df1.iloc[start_l[1], 0]

        def grow(self, new_l, df1):
            """线段生长 - 保持原始逻辑"""
            if self.d == 1:
                if df1.iloc[new_l[1], 1] >= self.cur_extreme:
                    if df1.iloc[new_l[0], 0] > self.prev_extreme:
                        self.gap = True
                    else:
                        self.gap = False
                    self.prev_extreme = self.cur_extreme
                    self.cur_extreme = df1.iloc[new_l[1], 1]
                    self.cur_extreme_pos = new_l[1]
                else:
                    if (self.gap == False and df1.iloc[new_l[1], 0] < df1.iloc[self.vertex[-1], 0]) or \
                            (self.gap == True and (df1.iloc[self.vertex[-1], 1] < df1.iloc[self.vertex[-3], 1]) and
                             (df1.iloc[self.vertex[-2], 0] < df1.iloc[self.vertex[-4], 0])):
                        self.finished = True
                        self.vertex = [i for i in self.vertex if i <= self.cur_extreme_pos]
                        return True, self.vertex[-1]

                self.vertex = self.vertex + new_l
                return False, 0
            else:
                if df1.iloc[new_l[1], 0] <= self.cur_extreme:
                    if df1.iloc[new_l[0], 1] < self.prev_extreme:
                        self.gap = True
                    else:
                        self.gap = False
                    self.vertex = self.vertex + new_l
                    self.prev_extreme = self.cur_extreme
                    self.cur_extreme = df1.iloc[new_l[1], 0]
                    self.cur_extreme_pos = new_l[1]
                else:
                    if (self.gap == False and df1.iloc[new_l[1], 1] > df1.iloc[self.vertex[-1], 1]) or \
                            (self.gap == True and (df1.iloc[self.vertex[-1], 0] > df1.iloc[self.vertex[-3], 0]) and
                             (df1.iloc[self.vertex[-2], 1] > df1.iloc[self.vertex[-4], 1])):
                        self.finished = True
                        self.vertex = [i for i in self.vertex if i <= self.cur_extreme_pos]
                        return True, self.vertex[-1]

                self.vertex = self.vertex + new_l
                return False, 0

        def lines(self, df1):
            """获取线段起点终点 - 保持原始逻辑"""
            if self.d == 1:
                return [(self.start, self.getrange(df1)[0]), (self.vertex[-1], self.getrange(df1)[1])]
            else:
                return [(self.start, self.getrange(df1)[0]), (self.vertex[-1], self.getrange(df1)[1])]

        def getrange(self, df1):
            """获取线段范围 - 保持原始逻辑"""
            if self.d == 1:
                return [df1.iloc[self.start, 0], self.cur_extreme, self.d]
            else:
                return [df1.iloc[self.start, 1], self.cur_extreme, self.d]

    def get_pivot(self, lines, df1):
        """获取中枢 - 保持原始逻辑"""
        Pivot1_array = []
        i = 0

        while i < len(lines):
            d = 2 * int(lines[i][0][1] < lines[i][1][1]) - 1
            if i < len(lines) - 3:
                if d == 1:
                    if lines[i + 3][1][1] <= lines[i + 1][0][1]:
                        pivot = self.Pivot1(lines[i:i + 4], d, df1)
                        i_j = 1
                        while i + i_j < len(lines) - 3 and pivot.finished == 0:
                            pivot.grow(lines[i + i_j + 3], df1)
                            i_j += 1

                        i = i + pivot.size
                        Pivot1_array.append(pivot)
                        continue
                    else:
                        i += 1
                else:
                    if lines[i + 3][1][1] >= lines[i + 1][0][1]:
                        pivot = self.Pivot1(lines[i:i + 4], d, df1)
                        i_j = 1
                        while i + i_j < len(lines) - 3 and pivot.finished == 0:
                            pivot.grow(lines[i + i_j + 3], df1)
                            i_j += 1

                        i = i + pivot.size
                        Pivot1_array.append(pivot)
                        continue
                    else:
                        i += 1
            else:
                i += 1

        tails = [df1.iloc[lines[-1][0][0], 3], lines[-1][0][1],
                 df1.iloc[lines[-1][1][0], 3], lines[-1][1][1],
                 2 * int(lines[-1][1][1] > lines[-1][0][1]) - 1]

        return Pivot1_array, tails

    class Pivot1:
        """中枢类 - 保持原始逻辑"""

        def __init__(self, lines, d, df1):
            self.trend = -2
            self.level = 1
            self.enter_d = d
            self.aft_l_price = 0
            self.aft_l_time = '00'
            self.future_zd = -float('inf')
            self.future_zg = float('inf')

            if d == 1:
                if lines[3][1][1] <= lines[1][0][1]:
                    self.zg = min(lines[1][0][1], lines[3][0][1])
                    self.zd = max(lines[3][1][1], lines[1][1][1])
                    self.dd = lines[2][0][1]
                    self.gg = max(lines[1][0][1], lines[2][1][1])
            else:
                if lines[3][1][1] >= lines[1][0][1]:
                    self.zg = min(lines[1][1][1], lines[3][1][1])
                    self.zd = max(lines[3][0][1], lines[1][0][1])
                    self.dd = min(lines[2][1][1], lines[1][0][1])
                    self.gg = lines[2][0][1]

            self.start_index = lines[1][0][0]
            self.end_index = lines[2][1][0]
            self.finished = 0
            self.enter_force = self.seg_force(lines[0])
            self.leave_force = self.seg_force(lines[3])
            self.size = 3
            self.mean = 0.5 * (self.zd + self.zg)
            self.start_time = df1.iloc[self.start_index, 3]
            self.leave_start_time = df1.iloc[self.end_index, 3]
            self.leave_end_time = df1.iloc[lines[3][1][0], 3]
            self.leave_d = -d
            self.leave_start_price = lines[3][0][1]
            self.leave_end_price = lines[3][1][1]
            self.prev2_force = self.seg_force(lines[1])
            self.prev1_force = self.seg_force(lines[2])
            self.prev2_end_price = lines[1][1][1]
            self.df1 = df1  # 保存df1引用

        def seg_force(self, seg):
            """计算线段力度 - 保持原始逻辑"""
            return 1000 * abs(seg[1][1] / seg[0][1] - 1) / (seg[1][0] - seg[0][0])

        def grow(self, seg, df1):
            """中枢生长 - 保持原始逻辑"""
            self.prev2_force = self.prev1_force
            self.prev1_force = self.leave_force
            self.prev2_end_price = self.leave_start_price

            if seg[1][1] > seg[0][1]:
                if (seg[1][1] >= self.zd and seg[0][1] <= self.zg) and (self.size <= 28):
                    self.end_index = seg[0][0]
                    self.size += 1
                    self.dd = min(self.dd, seg[0][1])

                    self.leave_force = self.seg_force(seg)
                    self.leave_start_time = df1.iloc[self.end_index, 3]
                    self.leave_end_time = df1.iloc[seg[1][0], 3]
                    self.leave_d = 2 * int(seg[1][1] > seg[0][1]) - 1
                    self.leave_start_price = seg[0][1]
                    self.leave_end_price = seg[1][1]

                    if self.size in [4, 7, 10, 19, 28]:
                        self.future_zd = max(self.future_zd, self.dd)
                        self.future_zg = min(self.future_zg, self.gg)

                    if self.size in [10, 28]:
                        self.level += 1
                        self.zd = self.future_zd
                        self.zg = self.future_zg
                        self.future_zd = -float('inf')
                        self.future_zg = float('inf')
                else:
                    if (seg[1][1] >= self.zd and seg[0][1] <= self.zg):
                        self.dd = min(self.dd, seg[0][1])
                        self.finished = 0.5
                    else:
                        self.finished = 1

                    self.aft_l_price = seg[1][1]
                    self.aft_l_time = df1.iloc[seg[1][0], 3]
            else:
                if (seg[1][1] <= self.zg and seg[0][1] >= self.zd) and self.size <= 28:
                    self.end_index = seg[0][0]
                    self.size += 1
                    self.gg = max(self.gg, seg[0][1])

                    self.leave_force = self.seg_force(seg)
                    self.leave_start_time = df1.iloc[self.end_index, 3]
                    self.leave_end_time = df1.iloc[seg[1][0], 3]
                    self.leave_d = 2 * int(seg[1][1] > seg[0][1]) - 1
                    self.leave_start_price = seg[0][1]
                    self.leave_end_price = seg[1][1]

                    if self.size in [4, 7, 10, 19, 28]:
                        self.future_zd = max(self.future_zd, self.dd)
                        self.future_zg = min(self.future_zg, self.gg)

                    if self.size in [10, 28]:
                        self.level += 1
                        self.zd = self.future_zd
                        self.zg = self.future_zg
                        self.future_zd = -float('inf')
                        self.future_zg = float('inf')
                else:
                    if (seg[1][1] <= self.zg and seg[0][1] >= self.zd):
                        self.gg = max(self.gg, seg[0][1])
                        self.finished = 0.5
                    else:
                        self.finished = 1

                    self.aft_l_price = seg[1][1]
                    self.aft_l_time = df1.iloc[seg[1][0], 3]

        def write_out(self, filepath, extra='', df1=None):
            """写入输出文件 - 保持原始逻辑"""
            if df1 is None:
                df1 = self.df1

            with open(filepath, 'w') as f:
                f.write(f' zd: {self.zd} zg: {self.zg} dd: {self.dd} gg: {self.gg}\n')
                f.write(
                    f' leave_d: {self.leave_d} prev2_leave_force: {self.prev2_force} leave_force: {self.leave_force}\n')
                f.write(f' start_time: {self.start_time} leave_start_time: {self.leave_start_time}\n')
                f.write(f' leave_end_time: {self.leave_end_time} prev2_end_price: {self.prev2_end_price}\n')
                f.write(f' leave_end_price: {self.leave_end_price} size: {self.size}\n')
                f.write(f' finished: {self.finished} trend: {self.trend} level: {self.level}\n')
                if extra != '':
                    f.write(f' tails: {extra}\n')
                    f.write(f' now: {df1.iloc[-1]}\n')

    def process_pivot(self, pivot_array):
        """处理中枢 - 保持原始逻辑"""
        for i in range(len(pivot_array) - 1):
            if pivot_array[i].level == 1 and pivot_array[i + 1].level == 1:
                if pivot_array[i].dd > pivot_array[i + 1].gg:
                    pivot_array[i + 1].trend = -1
                else:
                    if pivot_array[i].gg < pivot_array[i + 1].dd:
                        pivot_array[i + 1].trend = 1
                    else:
                        pivot_array[i + 1].trend = 0
            else:
                if pivot_array[i].gg > pivot_array[i + 1].gg and pivot_array[i].dd > pivot_array[i + 1].dd:
                    pivot_array[i + 1].trend = -1
                else:
                    if pivot_array[i].gg < pivot_array[i + 1].gg and pivot_array[i].dd < pivot_array[i + 1].dd:
                        pivot_array[i + 1].trend = 1
                    else:
                        pivot_array[i + 1].trend = 0

        return pivot_array

    def buy_point1(self, pro_pivot, tails, df1):
        """一类买点 - 保持原始逻辑"""
        if len(pro_pivot) <= 3 or tails[4] == 1 or pro_pivot[-1].size >= 8 or pro_pivot[-1].finished != 0 or \
                df1.iloc[-1][0] / pro_pivot[-1].leave_end_price - 1 > 0 or df1.iloc[-1][0] > tails[3]:
            return False, 0
        else:
            if (pro_pivot[-1].prev2_end_price > pro_pivot[-1].leave_end_price) and \
                    (pro_pivot[-1].leave_start_time == tails[0]) and \
                    df1.iloc[-1][0] < pro_pivot[-1].dd and \
                    1.2 * pro_pivot[-1].leave_force < pro_pivot[-1].prev2_force and \
                    (pro_pivot[-1].dd > pro_pivot[-1].leave_end_price):
                return True, pro_pivot[-1].dd
            else:
                return False, 0

    def buy_point2(self, pro_pivot, tails, df1):
        """二类买点 - 保持原始逻辑"""
        if len(pro_pivot) <= 3 or tails[4] == 1 or pro_pivot[-1].size >= 8 or pro_pivot[-1].finished != 0 or \
                df1.iloc[-1][0] / pro_pivot[-1].leave_end_price - 1 > 0 or df1.iloc[-1][0] > tails[3]:
            return False, 0
        else:
            if (pro_pivot[-1].prev2_end_price < pro_pivot[-1].leave_end_price) and \
                    (pro_pivot[-1].leave_start_time == tails[0]) and \
                    pro_pivot[-1].prev2_end_price == pro_pivot[-1].dd and \
                    pro_pivot[-1].leave_start_price > 0.51 * (pro_pivot[-1].zd + pro_pivot[-1].zg):
                return True, pro_pivot[-1].prev2_end_price
            else:
                return False, 0

    def buy_point3_des(self, pro_pivot, tails, df1):
        """三类买点（下降中枢） - 保持原始逻辑"""
        if len(pro_pivot) <= 2 or tails[4] == 1 or pro_pivot[-1].finished != 1 or \
                pro_pivot[-1].level > 1 or df1.iloc[-1][0] / pro_pivot[-1].leave_end_price - 1 > 0 or \
                df1.iloc[-1][0] > tails[3]:
            return False, 0
        else:
            if df1.iloc[-1][0] < 0.98 * pro_pivot[-1].leave_end_price and df1.iloc[-1][0] > 1.02 * pro_pivot[-1].zg and \
                    pro_pivot[-1].aft_l_price > 1.02 * pro_pivot[-1].zg and \
                    tails[0] == pro_pivot[-1].leave_end_time and \
                    pro_pivot[-1].leave_force > pro_pivot[-1].prev2_force and \
                    pro_pivot[-1].leave_end_price > pro_pivot[-1].prev2_end_price:
                return True, pro_pivot[-1].zg
            else:
                return False, 0

    def buy_point3(self, pro_pivot, tails, df1):
        """三类买点(3买点) - 标准的三类买点逻辑"""
        if len(pro_pivot) < 3 or tails[4] != 1 or pro_pivot[-1].finished != 1:
            return False, 0

        # 标准三类买点条件：
        # 1. 当前中枢是上升中枢
        # 2. 回调不跌破中枢上沿(ZG)
        # 3. 回调后再次上涨
        if pro_pivot[-1].trend == 1:  # 上升中枢
            if tails[3] > pro_pivot[-1].zg and df1.iloc[-1][0] > tails[3]:
                # 计算回调幅度
                pullback_ratio = (pro_pivot[-1].gg - tails[3]) / (pro_pivot[-1].gg - pro_pivot[-1].dd)
                if 0.2 < pullback_ratio < 0.8:  # 回调幅度在20%-80%之间
                    return True, pro_pivot[-1].zg  # 返回支撑位

        return False, 0

    def buy_point23(self, pro_pivot, tails, df1):
        """二三类买点重合 - 保持原始逻辑"""
        if len(pro_pivot) <= 3 or pro_pivot[-1].finished != 1 or \
                pro_pivot[-1].level > 1 or df1.iloc[-1][0] / pro_pivot[-1].leave_end_price - 1 > 0 or \
                df1.iloc[-1][0] > tails[3]:
            return False, 0
        else:
            if df1.iloc[-1][0] < 0.98 * pro_pivot[-1].leave_end_price and df1.iloc[-1][0] > 1.01 * pro_pivot[-1].zg and \
                    pro_pivot[-1].trend == -1 and tails[3] > 1.01 * pro_pivot[-1].zg and \
                    tails[0] == pro_pivot[-1].leave_end_time and \
                    pro_pivot[-1].leave_start_price == pro_pivot[-1].dd:
                return True, pro_pivot[-1].zg
            else:
                return False, 0

    def sell_point1(self, pro_pivot, tails, df1):
        """一类卖点 - 保持原始逻辑"""
        if len(pro_pivot) <= 3 or tails[4] == -1 or pro_pivot[-1].size >= 8 or pro_pivot[-1].finished != 0 or \
                df1.iloc[-1][1] / pro_pivot[-1].leave_end_price - 1 < 0 or df1.iloc[-1][0] < tails[3]:
            return False, 0
        else:
            if (pro_pivot[-1].prev2_end_price < pro_pivot[-1].leave_end_price) and \
                    (pro_pivot[-1].leave_start_time == tails[0]) and \
                    df1.iloc[-1][0] > pro_pivot[-1].zg and \
                    1.2 * pro_pivot[-1].leave_force < pro_pivot[-1].prev2_force:
                return True, pro_pivot[-1].zg
            else:
                return False, 0

    def sell_point2(self, pro_pivot, tails, df1):
        """二类卖点 - 保持原始逻辑"""
        if len(pro_pivot) <= 3 or tails[4] == -1 or pro_pivot[-1].size >= 8 or pro_pivot[-1].finished != 0 or \
                df1.iloc[-1][1] / pro_pivot[-1].leave_end_price - 1 < 0 or df1.iloc[-1][0] < tails[3]:
            return False, 0
        else:
            if (pro_pivot[-1].prev2_end_price > pro_pivot[-1].leave_end_price) and \
                    (pro_pivot[-1].leave_start_time == tails[0]) and \
                    df1.iloc[-1][0] > 0.51 * (pro_pivot[-1].zd + pro_pivot[-1].zg) and \
                    pro_pivot[-1].prev2_end_price == pro_pivot[-1].gg:
                return True, pro_pivot[-1].zg
            else:
                return False, 0

    def sell_point3_ris(self, pro_pivot, tails, df1):
        """三类卖点（上升中枢） - 保持原始逻辑"""
        if len(pro_pivot) <= 3 or tails[4] == -1 or pro_pivot[-1].size >= 8 or pro_pivot[-1].finished != 1 or \
                df1.iloc[-1][0] < tails[3]:
            return False, 0
        else:
            if (1.02 * pro_pivot[-1].leave_end_price < df1.iloc[-1][0]) and \
                    (pro_pivot[-1].leave_end_time == tails[0]) and \
                    pro_pivot[-1].leave_force > pro_pivot[-1].prev2_force and \
                    df1.iloc[-1][1] < pro_pivot[-1].zd:
                return True, pro_pivot[-1].zd
            else:
                return False, 0

    def seg_buy(self, lines, df1):
        """线段买点 - 保持原始逻辑"""
        if len(lines) <= 4 or df1.iloc[-1].iloc[0] > lines[-1][1][1]:
            return False, 0
        else:
            if lines[-1][1][1] < lines[-1][0][1] and ((1.2 * self.seg_force(lines[-1]) < self.seg_force(lines[-3]) and
                                                       lines[-1][1][1] < lines[-3][1][1]) or lines[-1][1][1] > 1.02 *
                                                      lines[-3][1][1]):
                return True, lines[-3][1][1]
            else:
                return False, 0

    def seg_sell(self, lines, df1):
        """线段卖点 - 保持原始逻辑"""
        if len(lines) <= 4 or df1.iloc[-1].iloc[0] < lines[-1][1][1]:
            return False, 0
        else:
            if lines[-1][1][1] > lines[-1][0][1] and ((1.2 * self.seg_force(lines[-1]) < self.seg_force(lines[-3]) and
                                                       lines[-1][1][1] > lines[-3][1][1]) or lines[-1][1][1] < 0.98 *
                                                      lines[-3][1][1]):
                return True, lines[-3][1][1]
            else:
                return False, 0

    def seg_force(self, seg):
        """计算线段力度 - 保持原始逻辑"""
        return 1000 * abs(seg[1][1] / seg[0][1] - 1) / (seg[1][0] - seg[0][0])

    def write_seg(self, temp_lines, file, buy_sign, interval, df1):
        """写入线段信号 - 保持原始逻辑"""
        with open(file, 'w') as f:
            if buy_sign:
                f.write(f'seg-3:{df1.iloc[temp_lines[-3][0][0], 3]} {df1.iloc[temp_lines[-3][0][0], 1]}')
                f.write(f'{df1.iloc[temp_lines[-3][1][0], 3]} {df1.iloc[temp_lines[-3][1][0], 0]}\n')
                f.write(f'seg-2:{df1.iloc[temp_lines[-2][0][0], 3]} {df1.iloc[temp_lines[-2][0][0], 0]}')
                f.write(f'{df1.iloc[temp_lines[-2][1][0], 3]} {df1.iloc[temp_lines[-2][1][0], 1]}\n')
                f.write(f'seg-1:{df1.iloc[temp_lines[-1][0][0], 3]} {df1.iloc[temp_lines[-1][0][0], 1]}')
                f.write(f'{df1.iloc[temp_lines[-1][1][0], 3]} {df1.iloc[temp_lines[-1][1][0], 0]}\n')
                f.write(f'cur_price:\n{df1.iloc[-1, 0]}\n')
                f.write(f'cur_time:\n{df1.iloc[-1, 3]}\n')
                if df1.iloc[temp_lines[-1][1][0], 0] < interval:
                    f.write(f'target_price:{interval}\n')
                else:
                    f.write(f'supp_price:{interval}\n')
            else:
                f.write(f'seg-3:{df1.iloc[temp_lines[-3][0][0], 3]} {df1.iloc[temp_lines[-3][0][0], 0]}')
                f.write(f'{df1.iloc[temp_lines[-3][1][0], 3]} {df1.iloc[temp_lines[-3][1][0], 1]}\n')
                f.write(f'seg-2:{df1.iloc[temp_lines[-2][0][0], 3]} {df1.iloc[temp_lines[-2][0][0], 1]}')
                f.write(f'{df1.iloc[temp_lines[-2][1][0], 3]} {df1.iloc[temp_lines[-2][1][0], 0]}\n')
                f.write(f'seg-1:{df1.iloc[temp_lines[-1][0][0], 3]} {df1.iloc[temp_lines[-1][0][0], 0]}')
                f.write(f'{df1.iloc[temp_lines[-1][1][0], 3]} {df1.iloc[temp_lines[-1][1][0], 1]}\n')
                f.write(f'cur_price:\n{df1.iloc[-1, 1]}\n')
                f.write(f'cur_time:\n{df1.iloc[-1, 3]}\n')
                if df1.iloc[temp_lines[-1][1][0], 0] > interval:
                    f.write(f'target_price:{interval}\n')
                else:
                    f.write(f'resist_price:{interval}\n')


# 主函数
if __name__ == "__main__":
    # 参数设置
    data_dir = "sp500_data"  # CSV文件所在目录
    output_base_dir = "tradingpoint"  # 输出目录

    # 创建分析实例并处理所有文件
    analyzer = ChanAnalysis(data_dir, output_base_dir, debug=16)
    analyzer.process_all_files()
    print("运行完毕")