#!/usr/bin/env python3
"""
候选号码生成模型 / Candidate Number Generation Model

基于统计分析结果和配置权重生成候选彩票号码：
- 权重计算 / Weight calculation
- 约束验证 / Constraint validation  
- 候选生成 / Candidate generation
- 理由说明 / Reasoning explanation
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Any
import math
from collections import Counter


class LotteryModel:
    """彩票号码生成模型类 / Lottery number generation model class"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化模型
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.front_range = tuple(config['analysis']['front_range'])
        self.back_range = tuple(config['analysis']['back_range'])
        self.metrics_weights = config['metrics']
        self.constraints = config['constraints']
        
    def calculate_number_weights(self, metrics_data: Dict[str, Any], bias: str) -> Dict[str, Dict[int, float]]:
        """
        基于统计指标计算号码权重
        
        Args:
            metrics_data: 统计指标数据
            bias: 偏向类型 ('hot', 'cold', 'neutral')
            
        Returns:
            包含前区和后区号码权重的字典
        """
        front_weights = {}
        back_weights = {}
        
        # 初始化基础权重
        for num in range(self.front_range[0], self.front_range[1] + 1):
            front_weights[num] = 1.0
        for num in range(self.back_range[0], self.back_range[1] + 1):
            back_weights[num] = 1.0
            
        # 1. 冷热号权重调整
        if 'front_z_scores' in metrics_data and 'back_z_scores' in metrics_data:
            self._apply_hot_cold_weights(front_weights, back_weights, 
                                       metrics_data['front_z_scores'], 
                                       metrics_data['back_z_scores'], bias)
        
        # 2. 遗漏值权重调整
        if 'missing_values' in metrics_data:
            self._apply_missing_weights(front_weights, back_weights, 
                                      metrics_data['missing_values'])
        
        # 3. 关联度权重调整
        if 'correlation_matrix' in metrics_data:
            self._apply_correlation_weights(front_weights, back_weights,
                                          metrics_data['correlation_matrix'])
                                          
        return {
            'front': front_weights,
            'back': back_weights
        }
    
    def _apply_hot_cold_weights(self, front_weights: Dict[int, float], back_weights: Dict[int, float],
                               front_z_scores: Dict[int, float], back_z_scores: Dict[int, float], 
                               bias: str) -> None:
        """应用冷热号权重调整"""
        weight_factor = self.metrics_weights['hot_cold_weight']
        
        # 前区号码权重调整
        for num, z_score in front_z_scores.items():
            if bias == "hot":
                # 热号偏向：提高正Z分数号码权重
                adjustment = max(0.1, 1 + z_score * weight_factor)
            elif bias == "cold":
                # 冷号偏向：提高负Z分数号码权重
                adjustment = max(0.1, 1 - z_score * weight_factor)
            else:
                # 中性：轻微调整
                adjustment = max(0.1, 1 + z_score * weight_factor * 0.5)
            front_weights[num] *= adjustment
            
        # 后区号码权重调整
        for num, z_score in back_z_scores.items():
            if bias == "hot":
                adjustment = max(0.1, 1 + z_score * weight_factor)
            elif bias == "cold":
                adjustment = max(0.1, 1 - z_score * weight_factor)
            else:
                adjustment = max(0.1, 1 + z_score * weight_factor * 0.5)
            back_weights[num] *= adjustment
    
    def _apply_missing_weights(self, front_weights: Dict[int, float], back_weights: Dict[int, float],
                              missing_values: Dict[str, Dict[int, int]]) -> None:
        """应用遗漏值权重调整"""
        weight_factor = self.metrics_weights['missing_weight']
        
        # 前区遗漏值调整
        if 'front' in missing_values:
            max_missing = max(missing_values['front'].values()) if missing_values['front'] else 1
            for num, missing in missing_values['front'].items():
                # 遗漏期数越多，权重越高
                adjustment = 1 + (missing / max_missing) * weight_factor
                front_weights[num] *= adjustment
                
        # 后区遗漏值调整
        if 'back' in missing_values:
            max_missing = max(missing_values['back'].values()) if missing_values['back'] else 1
            for num, missing in missing_values['back'].items():
                adjustment = 1 + (missing / max_missing) * weight_factor
                back_weights[num] *= adjustment
    
    def _apply_correlation_weights(self, front_weights: Dict[int, float], back_weights: Dict[int, float],
                                  correlation_matrix: Dict[str, np.ndarray]) -> None:
        """应用关联度权重调整"""
        weight_factor = self.metrics_weights['correlation_weight']
        
        # 前区关联度调整 - 降低高相关号码的权重，增加多样性
        if 'front' in correlation_matrix:
            corr_matrix = correlation_matrix['front']
            for i in range(len(corr_matrix)):
                num = i + self.front_range[0]
                if num in front_weights:
                    # 计算该号码与其他号码的平均相关度
                    avg_corr = np.mean(np.abs(corr_matrix[i]))
                    # 高相关度的号码权重稍微降低
                    adjustment = 1 - avg_corr * weight_factor
                    front_weights[num] *= max(0.1, adjustment)
                    
        # 后区关联度调整
        if 'back' in correlation_matrix:
            corr_matrix = correlation_matrix['back']
            for i in range(len(corr_matrix)):
                num = i + self.back_range[0]
                if num in back_weights:
                    avg_corr = np.mean(np.abs(corr_matrix[i]))
                    adjustment = 1 - avg_corr * weight_factor
                    back_weights[num] *= max(0.1, adjustment)
    
    def generate_candidates(self, weights: Dict[str, Dict[int, float]], 
                           num_candidates: int = 5) -> List[Dict[str, Any]]:
        """
        生成候选号码
        
        Args:
            weights: 号码权重
            num_candidates: 候选数量
            
        Returns:
            候选号码列表，每个包含号码和生成理由
        """
        candidates = []
        
        for i in range(num_candidates):
            # 生成一注候选号码
            candidate = self._generate_single_candidate(weights, i)
            if candidate:
                candidates.append(candidate)
                
        return candidates
    
    def _generate_single_candidate(self, weights: Dict[str, Dict[int, float]], 
                                  candidate_id: int) -> Dict[str, Any]:
        """
        生成单注候选号码
        
        Args:
            weights: 号码权重
            candidate_id: 候选编号
            
        Returns:
            单注候选号码及理由
        """
        max_attempts = 1000
        
        for attempt in range(max_attempts):
            # 生成前区号码
            front_numbers = self._select_weighted_numbers(
                weights['front'], 5, self.front_range
            )
            
            # 生成后区号码
            back_numbers = self._select_weighted_numbers(
                weights['back'], 2, self.back_range
            )
            
            # 验证约束条件
            if self._validate_constraints(front_numbers, back_numbers):
                # 生成理由说明
                reasoning = self._generate_reasoning(front_numbers, back_numbers, weights)
                
                return {
                    'id': candidate_id + 1,
                    'front_numbers': sorted(front_numbers),
                    'back_numbers': sorted(back_numbers),
                    'reasoning': reasoning,
                    'attempt': attempt + 1
                }
        
        # 如果无法满足约束，返回基础随机选择
        front_numbers = sorted(random.sample(range(self.front_range[0], self.front_range[1] + 1), 5))
        back_numbers = sorted(random.sample(range(self.back_range[0], self.back_range[1] + 1), 2))
        
        return {
            'id': candidate_id + 1,
            'front_numbers': front_numbers,
            'back_numbers': back_numbers,
            'reasoning': "基于随机选择生成（未能满足所有约束条件）",
            'attempt': max_attempts
        }
    
    def _select_weighted_numbers(self, weights: Dict[int, float], count: int, 
                                num_range: Tuple[int, int]) -> List[int]:
        """
        根据权重选择号码
        
        Args:
            weights: 号码权重字典
            count: 选择数量
            num_range: 号码范围
            
        Returns:
            选择的号码列表
        """
        numbers = list(range(num_range[0], num_range[1] + 1))
        weight_values = [weights.get(num, 1.0) for num in numbers]
        
        # 使用权重随机选择
        selected = []
        remaining_numbers = numbers.copy()
        remaining_weights = weight_values.copy()
        
        for _ in range(count):
            if not remaining_numbers:
                break
                
            # 权重随机选择
            chosen_idx = random.choices(range(len(remaining_numbers)), 
                                       weights=remaining_weights, k=1)[0]
            selected.append(remaining_numbers[chosen_idx])
            
            # 移除已选择的号码
            remaining_numbers.pop(chosen_idx)
            remaining_weights.pop(chosen_idx)
            
        return selected
    
    def _validate_constraints(self, front_numbers: List[int], back_numbers: List[int]) -> bool:
        """
        验证候选号码是否满足约束条件
        
        Args:
            front_numbers: 前区号码
            back_numbers: 后区号码
            
        Returns:
            是否满足约束条件
        """
        # 1. 和值约束
        if not self._check_sum_constraints(front_numbers):
            return False
            
        # 2. 奇偶比约束
        if not self._check_odd_even_constraints(front_numbers, back_numbers):
            return False
            
        # 3. 区间分布约束
        if not self._check_range_distribution_constraints(front_numbers):
            return False
            
        return True
    
    def _check_sum_constraints(self, front_numbers: List[int]) -> bool:
        """检查和值约束"""
        if 'sum_range' not in self.constraints:
            return True
            
        front_sum = sum(front_numbers)
        sum_range = self.constraints['sum_range']
        
        return sum_range['front_min'] <= front_sum <= sum_range['front_max']
    
    def _check_odd_even_constraints(self, front_numbers: List[int], back_numbers: List[int]) -> bool:
        """检查奇偶比约束"""
        if 'odd_even_ratio' not in self.constraints:
            return True
            
        ratio_config = self.constraints['odd_even_ratio']
        
        # 前区奇数个数
        front_odd_count = sum(1 for num in front_numbers if num % 2 == 1)
        if not (ratio_config['front_min_odd'] <= front_odd_count <= ratio_config['front_max_odd']):
            return False
            
        # 后区奇数个数
        back_odd_count = sum(1 for num in back_numbers if num % 2 == 1)
        if not (ratio_config['back_min_odd'] <= back_odd_count <= ratio_config['back_max_odd']):
            return False
            
        return True
    
    def _check_range_distribution_constraints(self, front_numbers: List[int]) -> bool:
        """检查区间分布约束"""
        if 'range_distribution' not in self.constraints or 'front_ranges' not in self.constraints['range_distribution']:
            return True
            
        ranges = self.constraints['range_distribution']['front_ranges']
        
        for range_def in ranges:
            count = sum(1 for num in front_numbers 
                       if range_def['min'] <= num <= range_def['max'])
            if not (range_def['min_count'] <= count <= range_def['max_count']):
                return False
                
        return True
    
    def _generate_reasoning(self, front_numbers: List[int], back_numbers: List[int], 
                           weights: Dict[str, Dict[int, float]]) -> str:
        """
        生成选号理由说明
        
        Args:
            front_numbers: 前区号码
            back_numbers: 后区号码
            weights: 号码权重
            
        Returns:
            理由说明文本
        """
        reasons = []
        
        # 1. 和值分析
        front_sum = sum(front_numbers)
        reasons.append(f"前区和值{front_sum}，处于合理范围")
        
        # 2. 奇偶比分析
        front_odd = sum(1 for num in front_numbers if num % 2 == 1)
        front_even = len(front_numbers) - front_odd
        back_odd = sum(1 for num in back_numbers if num % 2 == 1)
        back_even = len(back_numbers) - back_odd
        reasons.append(f"前区奇偶比{front_odd}:{front_even}，后区奇偶比{back_odd}:{back_even}")
        
        # 3. 权重分析
        front_avg_weight = np.mean([weights['front'].get(num, 1.0) for num in front_numbers])
        back_avg_weight = np.mean([weights['back'].get(num, 1.0) for num in back_numbers])
        
        if front_avg_weight > 1.2:
            reasons.append("前区号码权重较高，符合统计趋势")
        elif front_avg_weight < 0.8:
            reasons.append("前区选择冷门号码，遵循均值回归")
        else:
            reasons.append("前区号码权重平衡")
            
        if back_avg_weight > 1.2:
            reasons.append("后区号码权重较高，符合统计趋势")
        elif back_avg_weight < 0.8:
            reasons.append("后区选择冷门号码，遵循均值回归")
        else:
            reasons.append("后区号码权重平衡")
        
        # 4. 区间分布分析
        if 'range_distribution' in self.constraints and 'front_ranges' in self.constraints['range_distribution']:
            ranges = self.constraints['range_distribution']['front_ranges']
            range_dist = []
            for range_def in ranges:
                count = sum(1 for num in front_numbers 
                           if range_def['min'] <= num <= range_def['max'])
                if count > 0:
                    range_dist.append(f"{range_def['min']}-{range_def['max']}区间{count}个")
            if range_dist:
                reasons.append(f"区间分布：{', '.join(range_dist)}")
        
        # 5. 连号分析
        front_consecutive = 0
        sorted_front = sorted(front_numbers)
        for i in range(len(sorted_front) - 1):
            if sorted_front[i+1] - sorted_front[i] == 1:
                front_consecutive += 1
        
        if front_consecutive > 0:
            reasons.append(f"包含{front_consecutive}组连号")
        else:
            reasons.append("无连号，分布较散")
            
        return "；".join(reasons)