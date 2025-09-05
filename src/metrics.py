#!/usr/bin/env python3
"""
统计指标计算模块 / Statistical Metrics Calculation Module

提供彩票历史数据的各种统计指标计算功能：
- 冷热号分析 / Hot/cold number analysis
- 遗漏值计算 / Missing value calculation  
- 和值统计 / Sum value statistics
- 奇偶比分析 / Odd/even ratio analysis
- 区间分布分析 / Range distribution analysis
- 连号分析 / Consecutive numbers analysis
- 同尾分析 / Same tail analysis
- 关联度分析 / Correlation analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import math
from collections import Counter, defaultdict
from scipy import stats


class LotteryMetrics:
    """彩票统计指标计算类 / Lottery statistical metrics calculation class"""
    
    def __init__(self, front_range: Tuple[int, int], back_range: Tuple[int, int]):
        """
        初始化指标计算器
        
        Args:
            front_range: 前区号码范围 (min, max)
            back_range: 后区号码范围 (min, max)
        """
        self.front_range = front_range
        self.back_range = back_range
        
    def calculate_frequency(self, draws: List[Dict], recent_window: int = None) -> Dict[str, Dict[int, int]]:
        """
        计算号码频率统计
        
        Args:
            draws: 开奖数据列表
            recent_window: 最近窗口大小，None表示全部数据
            
        Returns:
            包含前区和后区频率统计的字典
        """
        if recent_window:
            draws = draws[-recent_window:]
            
        front_freq = Counter()
        back_freq = Counter()
        
        for draw in draws:
            # 前区号码
            if isinstance(draw['front_numbers'], list):
                for num in draw['front_numbers']:
                    front_freq[num] += 1
            
            # 后区号码  
            if isinstance(draw['back_numbers'], list):
                for num in draw['back_numbers']:
                    back_freq[num] += 1
                    
        return {
            'front': dict(front_freq),
            'back': dict(back_freq)
        }
    
    def calculate_z_scores(self, baseline_freq: Dict[int, int], recent_freq: Dict[int, int],
                          total_baseline: int, total_recent: int, num_range: Tuple[int, int]) -> Dict[int, float]:
        """
        计算Z分数进行趋势分析
        
        Args:
            baseline_freq: 基线频率统计
            recent_freq: 最近频率统计
            total_baseline: 基线总数
            total_recent: 最近总数
            num_range: 号码范围
            
        Returns:
            每个号码的Z分数字典
        """
        z_scores = {}
        
        for num in range(num_range[0], num_range[1] + 1):
            baseline_count = baseline_freq.get(num, 0)
            recent_count = recent_freq.get(num, 0)
            
            # 计算概率
            p_base = baseline_count / total_baseline if total_baseline > 0 else 0
            p_recent = recent_count / total_recent if total_recent > 0 else 0
            
            # 计算Z分数，处理极端情况
            if p_base == 0 or p_base == 1:
                p_base_safe = max(0.001, min(0.999, p_base))
            else:
                p_base_safe = p_base
                
            denominator = math.sqrt(p_base_safe * (1 - p_base_safe) / total_recent)
            
            if denominator > 0:
                z_score = (p_recent - p_base) / denominator
            else:
                z_score = 0.0
                
            z_scores[num] = z_score
            
        return z_scores
    
    def calculate_missing_values(self, draws: List[Dict]) -> Dict[str, Dict[int, int]]:
        """
        计算号码遗漏值（自上次出现以来的期数）
        
        Args:
            draws: 开奖数据列表（按时间倒序）
            
        Returns:
            包含前区和后区遗漏值的字典
        """
        front_missing = {}
        back_missing = {}
        
        # 初始化所有号码的遗漏值
        for num in range(self.front_range[0], self.front_range[1] + 1):
            front_missing[num] = len(draws)
        for num in range(self.back_range[0], self.back_range[1] + 1):
            back_missing[num] = len(draws)
            
        # 从最新开奖往前查找，记录每个号码的遗漏期数
        for period, draw in enumerate(draws):
            # 前区号码
            if isinstance(draw['front_numbers'], list):
                for num in draw['front_numbers']:
                    if num in front_missing and front_missing[num] == len(draws):
                        front_missing[num] = period
                        
            # 后区号码
            if isinstance(draw['back_numbers'], list):
                for num in draw['back_numbers']:
                    if num in back_missing and back_missing[num] == len(draws):
                        back_missing[num] = period
                        
        return {
            'front': front_missing,
            'back': back_missing
        }
    
    def calculate_sum_distribution(self, draws: List[Dict]) -> Dict[str, List[int]]:
        """
        计算和值分布统计
        
        Args:
            draws: 开奖数据列表
            
        Returns:
            包含前区和后区和值列表的字典
        """
        front_sums = []
        back_sums = []
        
        for draw in draws:
            if isinstance(draw['front_numbers'], list):
                front_sums.append(sum(draw['front_numbers']))
            if isinstance(draw['back_numbers'], list):
                back_sums.append(sum(draw['back_numbers']))
                
        return {
            'front': front_sums,
            'back': back_sums
        }
    
    def calculate_odd_even_ratio(self, draws: List[Dict]) -> Dict[str, List[Dict]]:
        """
        计算奇偶比分布
        
        Args:
            draws: 开奖数据列表
            
        Returns:
            包含前区和后区奇偶比统计的字典
        """
        front_ratios = []
        back_ratios = []
        
        for draw in draws:
            # 前区奇偶比
            if isinstance(draw['front_numbers'], list):
                odd_count = sum(1 for num in draw['front_numbers'] if num % 2 == 1)
                even_count = len(draw['front_numbers']) - odd_count
                front_ratios.append({'odd': odd_count, 'even': even_count})
                
            # 后区奇偶比
            if isinstance(draw['back_numbers'], list):
                odd_count = sum(1 for num in draw['back_numbers'] if num % 2 == 1)
                even_count = len(draw['back_numbers']) - odd_count
                back_ratios.append({'odd': odd_count, 'even': even_count})
                
        return {
            'front': front_ratios,
            'back': back_ratios
        }
    
    def calculate_range_distribution(self, draws: List[Dict], ranges: List[Dict]) -> Dict[str, List[Dict]]:
        """
        计算区间分布统计
        
        Args:
            draws: 开奖数据列表
            ranges: 区间定义列表，每个元素包含 {min, max, min_count, max_count}
            
        Returns:
            包含每期各区间号码分布的字典
        """
        distributions = []
        
        for draw in draws:
            if isinstance(draw['front_numbers'], list):
                range_counts = []
                for range_def in ranges:
                    count = sum(1 for num in draw['front_numbers'] 
                              if range_def['min'] <= num <= range_def['max'])
                    range_counts.append({
                        'range': f"{range_def['min']}-{range_def['max']}",
                        'count': count
                    })
                distributions.append(range_counts)
                
        return {'front': distributions}
    
    def calculate_consecutive_numbers(self, draws: List[Dict]) -> Dict[str, List[int]]:
        """
        计算连号统计
        
        Args:
            draws: 开奖数据列表
            
        Returns:
            包含前区和后区连号个数的字典
        """
        front_consecutive = []
        back_consecutive = []
        
        for draw in draws:
            # 前区连号
            if isinstance(draw['front_numbers'], list):
                sorted_nums = sorted(draw['front_numbers'])
                consecutive_count = 0
                for i in range(len(sorted_nums) - 1):
                    if sorted_nums[i+1] - sorted_nums[i] == 1:
                        consecutive_count += 1
                front_consecutive.append(consecutive_count)
                
            # 后区连号
            if isinstance(draw['back_numbers'], list):
                sorted_nums = sorted(draw['back_numbers'])
                consecutive_count = 0
                for i in range(len(sorted_nums) - 1):
                    if sorted_nums[i+1] - sorted_nums[i] == 1:
                        consecutive_count += 1
                back_consecutive.append(consecutive_count)
                
        return {
            'front': front_consecutive,
            'back': back_consecutive
        }
    
    def calculate_same_tail_numbers(self, draws: List[Dict]) -> Dict[str, List[int]]:
        """
        计算同尾号统计
        
        Args:
            draws: 开奖数据列表
            
        Returns:
            包含前区和后区同尾号个数的字典
        """
        front_same_tail = []
        back_same_tail = []
        
        for draw in draws:
            # 前区同尾号
            if isinstance(draw['front_numbers'], list):
                tail_counts = Counter(num % 10 for num in draw['front_numbers'])
                same_tail_count = sum(1 for count in tail_counts.values() if count > 1)
                front_same_tail.append(same_tail_count)
                
            # 后区同尾号
            if isinstance(draw['back_numbers'], list):
                tail_counts = Counter(num % 10 for num in draw['back_numbers'])
                same_tail_count = sum(1 for count in tail_counts.values() if count > 1)
                back_same_tail.append(same_tail_count)
                
        return {
            'front': front_same_tail,
            'back': back_same_tail
        }
    
    def calculate_correlation_matrix(self, draws: List[Dict]) -> Dict[str, np.ndarray]:
        """
        计算号码间关联度矩阵
        
        Args:
            draws: 开奖数据列表
            
        Returns:
            包含前区和后区关联度矩阵的字典
        """
        # 构建号码出现矩阵
        front_matrix = []
        back_matrix = []
        
        for draw in draws:
            # 前区号码向量
            front_vector = [0] * (self.front_range[1] - self.front_range[0] + 1)
            if isinstance(draw['front_numbers'], list):
                for num in draw['front_numbers']:
                    idx = num - self.front_range[0]
                    if 0 <= idx < len(front_vector):
                        front_vector[idx] = 1
            front_matrix.append(front_vector)
            
            # 后区号码向量
            back_vector = [0] * (self.back_range[1] - self.back_range[0] + 1)
            if isinstance(draw['back_numbers'], list):
                for num in draw['back_numbers']:
                    idx = num - self.back_range[0]
                    if 0 <= idx < len(back_vector):
                        back_vector[idx] = 1
            back_matrix.append(back_vector)
        
        # 计算相关系数矩阵
        front_corr = np.corrcoef(np.array(front_matrix).T)
        back_corr = np.corrcoef(np.array(back_matrix).T)
        
        # 处理NaN值
        front_corr = np.nan_to_num(front_corr, nan=0.0)
        back_corr = np.nan_to_num(back_corr, nan=0.0)
        
        return {
            'front': front_corr,
            'back': back_corr
        }
    
    def determine_bias(self, front_z_scores: Dict[int, float], back_z_scores: Dict[int, float],
                      front_baseline_freq: Dict[int, int], front_recent_freq: Dict[int, int],
                      back_baseline_freq: Dict[int, int], back_recent_freq: Dict[int, int],
                      total_baseline: int, total_recent: int) -> str:
        """
        确定整体偏向（热/冷/中性）
        
        Args:
            front_z_scores: 前区Z分数
            back_z_scores: 后区Z分数
            front_baseline_freq: 前区基线频率
            front_recent_freq: 前区最近频率
            back_baseline_freq: 后区基线频率
            back_recent_freq: 后区最近频率
            total_baseline: 基线总数
            total_recent: 最近总数
            
        Returns:
            偏向类型: 'hot', 'cold', 'neutral'
        """
        # 计算所有号码的概率差值
        deltas = []
        z_values = []
        
        # 前区号码
        for num in range(self.front_range[0], self.front_range[1] + 1):
            baseline_prob = front_baseline_freq.get(num, 0) / total_baseline
            recent_prob = front_recent_freq.get(num, 0) / total_recent
            delta = recent_prob - baseline_prob
            deltas.append(delta)
            z_values.append(front_z_scores[num])
        
        # 后区号码
        for num in range(self.back_range[0], self.back_range[1] + 1):
            baseline_prob = back_baseline_freq.get(num, 0) / total_baseline
            recent_prob = back_recent_freq.get(num, 0) / total_recent
            delta = recent_prob - baseline_prob
            deltas.append(delta)
            z_values.append(back_z_scores[num])
        
        # 计算统计指标
        mean_delta = np.mean(deltas)
        high_z_count = sum(1 for z in z_values if abs(z) >= 1.0)
        high_z_ratio = high_z_count / len(z_values) if z_values else 0
        
        positive_z_count = sum(1 for z in z_values if z > 0)
        negative_z_count = sum(1 for z in z_values if z < 0)
        
        # 判断偏向
        if abs(mean_delta) < 0.002 and high_z_ratio < 0.55:
            return "neutral"
        elif positive_z_count > negative_z_count and mean_delta >= 0.002:
            return "hot"
        elif negative_z_count > positive_z_count and mean_delta <= -0.002:
            return "cold"
        else:
            return "neutral"