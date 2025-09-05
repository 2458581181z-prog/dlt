#!/usr/bin/env python3
"""
统计分析与候选号码生成框架主程序
Statistical Analysis and Candidate Number Generation Framework

基于历史开奖数据提取常见指标，通过可配置的权重与约束生成候选号码与理由说明
Extract common metrics from historical draw data, generate candidate numbers and explanations 
through configurable weights and constraints
"""

import sys
import os
import argparse
import yaml
import pandas as pd
import json
import datetime
from pathlib import Path
from typing import Dict, List, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.metrics import LotteryMetrics
from src.model import LotteryModel


class LotteryAnalysisFramework:
    """彩票分析框架主类 / Main lottery analysis framework class"""
    
    def __init__(self, config_path: str):
        """
        初始化分析框架
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self.load_config(config_path)
        self.metrics = LotteryMetrics(
            front_range=tuple(self.config['analysis']['front_range']),
            back_range=tuple(self.config['analysis']['back_range'])
        )
        self.model = LotteryModel(self.config)
        self.draws_data = []
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Error loading config file {config_path}: {e}")
            sys.exit(1)
    
    def load_data(self) -> None:
        """加载历史开奖数据"""
        input_file = self.config['data']['input_file']
        
        print(f"Loading data from {input_file}...")
        
        if not os.path.exists(input_file):
            print(f"Error: Data file {input_file} not found.")
            print("Please prepare your data and place it at the configured path.")
            print("You can also modify the path in config.yaml")
            sys.exit(1)
        
        try:
            # 检查文件格式
            if input_file.endswith('.csv'):
                self._load_csv_data(input_file)
            elif input_file.endswith('.json') or input_file.endswith('.txt'):
                self._load_json_data(input_file)
            else:
                print(f"Unsupported file format: {input_file}")
                print("Supported formats: .csv, .json, .txt")
                sys.exit(1)
                
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)
        
        print(f"Loaded {len(self.draws_data)} draws successfully")
        
    def _load_csv_data(self, file_path: str) -> None:
        """
        加载CSV格式数据
        
        Args:
            file_path: CSV文件路径
        """
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 获取列映射配置
        columns = self.config['data']['columns']
        
        # 验证必要列是否存在
        required_columns = [columns['date'], columns['front_numbers'], columns['back_numbers']]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns in CSV: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            print("Please check your column mapping in config.yaml")
            sys.exit(1)
        
        # 转换数据格式
        for _, row in df.iterrows():
            draw_data = {
                'date': str(row[columns['date']]),
                'draw_number': str(row.get(columns.get('draw_number', ''), '')),
                'front_numbers': self._parse_numbers(row[columns['front_numbers']]),
                'back_numbers': self._parse_numbers(row[columns['back_numbers']])
            }
            self.draws_data.append(draw_data)
        
        # 按日期排序（最新的在前）
        self.draws_data.sort(key=lambda x: x['date'], reverse=True)
        
        # 限制数据量
        total_draws = self.config['analysis']['total_draws']
        if len(self.draws_data) > total_draws:
            self.draws_data = self.draws_data[:total_draws]
    
    def _load_json_data(self, file_path: str) -> None:
        """
        加载JSON格式数据（兼容现有格式）
        
        Args:
            file_path: JSON文件路径
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析多个JSON对象
        json_objects = []
        current_obj = ''
        brace_count = 0
        
        for char in content:
            current_obj += char
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and current_obj.strip():
                    try:
                        obj = json.loads(current_obj.strip())
                        json_objects.append(obj)
                    except json.JSONDecodeError:
                        pass
                    current_obj = ''
        
        # 提取开奖数据
        all_draws = []
        for obj in json_objects:
            if 'value' in obj and 'list' in obj['value']:
                all_draws.extend(obj['value']['list'])
        
        # 转换数据格式
        for draw in all_draws:
            if 'lotteryDrawResult' in draw and 'lotteryDrawTime' in draw:
                numbers = draw['lotteryDrawResult'].split()
                if len(numbers) >= 7:  # 5前区 + 2后区
                    front_numbers = [int(x) for x in numbers[:5]]
                    back_numbers = [int(x) for x in numbers[5:7]]
                    
                    draw_data = {
                        'date': draw['lotteryDrawTime'],
                        'draw_number': draw.get('lotteryDrawNum', ''),
                        'front_numbers': front_numbers,
                        'back_numbers': back_numbers
                    }
                    self.draws_data.append(draw_data)
        
        # 限制数据量
        total_draws = self.config['analysis']['total_draws']
        if len(self.draws_data) > total_draws:
            self.draws_data = self.draws_data[:total_draws]
        
        # 反转为时间顺序（最老的在前）
        self.draws_data.reverse()
    
    def _parse_numbers(self, numbers_str: str) -> List[int]:
        """
        解析号码字符串
        
        Args:
            numbers_str: 号码字符串（可能用逗号、空格分隔）
            
        Returns:
            号码整数列表
        """
        if pd.isna(numbers_str):
            return []
        
        # 处理不同的分隔符
        numbers_str = str(numbers_str).replace(',', ' ').replace('，', ' ')
        numbers = []
        
        for num_str in numbers_str.split():
            try:
                num = int(num_str.strip())
                numbers.append(num)
            except ValueError:
                continue
                
        return numbers
    
    def run_analysis(self) -> Dict[str, Any]:
        """
        运行完整分析流程
        
        Returns:
            分析结果字典
        """
        print("Starting lottery analysis...")
        
        if not self.draws_data:
            print("Error: No valid draw data loaded")
            sys.exit(1)
        
        recent_window = self.config['analysis']['recent_window']
        
        # 1. 计算基线频率统计
        print(f"Calculating baseline frequency for {len(self.draws_data)} draws...")
        baseline_freq = self.metrics.calculate_frequency(self.draws_data)
        
        # 2. 计算最近趋势统计
        print(f"Calculating recent trends for last {recent_window} draws...")
        recent_freq = self.metrics.calculate_frequency(self.draws_data, recent_window)
        
        # 3. 计算Z分数
        print("Calculating z-scores...")
        total_baseline = len(self.draws_data) * 5  # 前区每期5个号码
        total_recent = min(recent_window, len(self.draws_data)) * 5
        
        front_z_scores = self.metrics.calculate_z_scores(
            baseline_freq['front'], recent_freq['front'],
            total_baseline, total_recent,
            self.metrics.front_range
        )
        
        total_baseline_back = len(self.draws_data) * 2  # 后区每期2个号码
        total_recent_back = min(recent_window, len(self.draws_data)) * 2
        
        back_z_scores = self.metrics.calculate_z_scores(
            baseline_freq['back'], recent_freq['back'],
            total_baseline_back, total_recent_back,
            self.metrics.back_range
        )
        
        # 4. 确定偏向
        print("Determining bias...")
        bias = self.metrics.determine_bias(
            front_z_scores, back_z_scores,
            baseline_freq['front'], recent_freq['front'],
            baseline_freq['back'], recent_freq['back'],
            total_baseline, total_recent
        )
        
        # 5. 计算其他统计指标
        print("Calculating additional metrics...")
        missing_values = self.metrics.calculate_missing_values(self.draws_data[::-1])  # 最新在前
        sum_distribution = self.metrics.calculate_sum_distribution(self.draws_data)
        odd_even_ratios = self.metrics.calculate_odd_even_ratio(self.draws_data)
        
        if 'range_distribution' in self.config['constraints']:
            range_distribution = self.metrics.calculate_range_distribution(
                self.draws_data, 
                self.config['constraints']['range_distribution']['front_ranges']
            )
        else:
            range_distribution = None
            
        consecutive_numbers = self.metrics.calculate_consecutive_numbers(self.draws_data)
        same_tail_numbers = self.metrics.calculate_same_tail_numbers(self.draws_data)
        correlation_matrix = self.metrics.calculate_correlation_matrix(self.draws_data)
        
        # 6. 整合分析结果
        analysis_results = {
            'bias': bias,
            'total_draws': len(self.draws_data),
            'recent_window': recent_window,
            'baseline_freq': baseline_freq,
            'recent_freq': recent_freq,
            'front_z_scores': front_z_scores,
            'back_z_scores': back_z_scores,
            'missing_values': missing_values,
            'sum_distribution': sum_distribution,
            'odd_even_ratios': odd_even_ratios,
            'range_distribution': range_distribution,
            'consecutive_numbers': consecutive_numbers,
            'same_tail_numbers': same_tail_numbers,
            'correlation_matrix': correlation_matrix,
            'analysis_date': datetime.datetime.now().isoformat()
        }
        
        print(f"Analysis completed. Detected bias: {bias.upper()}")
        return analysis_results
    
    def generate_candidates(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        生成候选号码
        
        Args:
            analysis_results: 分析结果
            
        Returns:
            候选号码列表
        """
        print("Generating candidate numbers...")
        
        # 计算号码权重
        weights = self.model.calculate_number_weights(analysis_results, analysis_results['bias'])
        
        # 生成候选号码
        num_candidates = self.config['output']['num_candidates']
        candidates = self.model.generate_candidates(weights, num_candidates)
        
        print(f"Generated {len(candidates)} candidate picks")
        return candidates
    
    def save_results(self, analysis_results: Dict[str, Any], candidates: List[Dict[str, Any]]) -> None:
        """
        保存分析结果和候选号码
        
        Args:
            analysis_results: 分析结果
            candidates: 候选号码列表
        """
        # 创建输出目录
        output_dir = Path(self.config['output']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.datetime.now().strftime(self.config['output']['timestamp_format'])
        
        # 保存候选号码CSV文件
        candidates_file = output_dir / f"{self.config['output']['candidates_prefix']}_{timestamp}.csv"
        self._save_candidates_csv(candidates, candidates_file)
        
        # 保存分析报告Markdown文件
        report_file = output_dir / f"{self.config['output']['report_prefix']}_{timestamp}.md"
        self._save_analysis_report(analysis_results, candidates, report_file)
        
        print(f"Results saved:")
        print(f"  - Candidates: {candidates_file}")
        print(f"  - Report: {report_file}")
    
    def _save_candidates_csv(self, candidates: List[Dict[str, Any]], file_path: Path) -> None:
        """
        保存候选号码为CSV格式
        
        Args:
            candidates: 候选号码列表
            file_path: 输出文件路径
        """
        # 准备CSV数据
        csv_data = []
        for candidate in candidates:
            front_str = ' '.join([f"{num:02d}" for num in candidate['front_numbers']])
            back_str = ' '.join([f"{num:02d}" for num in candidate['back_numbers']])
            
            csv_data.append({
                '候选编号': candidate['id'],
                '前区号码': front_str,
                '后区号码': back_str,
                '完整号码': f"{front_str} + {back_str}",
                '生成理由': candidate['reasoning'],
                '尝试次数': candidate.get('attempt', 1)
            })
        
        # 保存为CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
    
    def _save_analysis_report(self, analysis_results: Dict[str, Any], 
                             candidates: List[Dict[str, Any]], file_path: Path) -> None:
        """
        保存分析报告为Markdown格式
        
        Args:
            analysis_results: 分析结果
            candidates: 候选号码列表
            file_path: 输出文件路径
        """
        report_lines = []
        
        # 报告头部
        report_lines.extend([
            "# 超级大乐透统计分析报告",
            "",
            "**⚠️ 仅供娱乐参考 ⚠️**",
            "",
            "本分析仅供教育和娱乐目的。彩票游戏涉及财务风险，所有号码组合在公平彩票中理论上等概率。",
            "此工具不保证任何中奖概率提升。请理性投注，量力而行。",
            "",
            f"**分析时间：** {analysis_results['analysis_date']}",
            f"**数据集：** {analysis_results['total_draws']} 期历史开奖",
            f"**趋势窗口：** {analysis_results['recent_window']} 期",
            f"**检测偏向：** {analysis_results['bias'].upper()}",
            ""
        ])
        
        # 候选号码
        report_lines.extend([
            "## 生成的候选号码",
            ""
        ])
        
        for candidate in candidates:
            front_str = ' '.join([f"{num:02d}" for num in candidate['front_numbers']])
            back_str = ' '.join([f"{num:02d}" for num in candidate['back_numbers']])
            
            report_lines.extend([
                f"### 候选 {candidate['id']}: {front_str} + {back_str}",
                "",
                f"**生成理由：** {candidate['reasoning']}",
                "",
                f"*生成尝试次数：{candidate.get('attempt', 1)}*",
                ""
            ])
        
        # 统计摘要
        report_lines.extend([
            "## 统计分析摘要",
            "",
            "### 前区号码统计 (1-35)",
            ""
        ])
        
        # 前区热号
        front_z_sorted = sorted(analysis_results['front_z_scores'].items(), 
                               key=lambda x: x[1], reverse=True)
        
        hot_front = [(num, z) for num, z in front_z_sorted if z > 0][:10]
        if hot_front:
            report_lines.extend([
                "**前10热号（正Z分数）：**",
                ""
            ])
            for num, z in hot_front:
                baseline_freq = analysis_results['baseline_freq']['front'].get(num, 0)
                recent_freq = analysis_results['recent_freq']['front'].get(num, 0)
                report_lines.append(f"- {num:2d}: z={z:+.2f} (基线: {baseline_freq}, 最近: {recent_freq})")
            report_lines.append("")
        
        # 前区冷号
        cold_front = [(num, z) for num, z in front_z_sorted if z < 0][-10:]
        if cold_front:
            report_lines.extend([
                "**前10冷号（负Z分数）：**",
                ""
            ])
            for num, z in reversed(cold_front):
                baseline_freq = analysis_results['baseline_freq']['front'].get(num, 0)
                recent_freq = analysis_results['recent_freq']['front'].get(num, 0)
                report_lines.append(f"- {num:2d}: z={z:+.2f} (基线: {baseline_freq}, 最近: {recent_freq})")
            report_lines.append("")
        
        # 后区统计
        report_lines.extend([
            "### 后区号码统计 (1-12)",
            ""
        ])
        
        back_z_sorted = sorted(analysis_results['back_z_scores'].items(), 
                              key=lambda x: x[1], reverse=True)
        
        for num, z in back_z_sorted:
            baseline_freq = analysis_results['baseline_freq']['back'].get(num, 0)
            recent_freq = analysis_results['recent_freq']['back'].get(num, 0)
            report_lines.append(f"- {num:2d}: z={z:+.2f} (基线: {baseline_freq}, 最近: {recent_freq})")
        
        report_lines.extend([
            "",
            "## 分析方法论",
            "",
            "1. **基线分析：** 对所有历史开奖进行频率计算",
            "2. **趋势窗口：** 对最近N期开奖进行频率计算", 
            "3. **Z分数计算：** 衡量最近趋势相对基线的统计显著性",
            "4. **偏向检测：** 基于统计阈值自动分类为热/冷/中性",
            "5. **权重计算：** 综合多种指标计算号码选择权重",
            "6. **约束验证：** 确保生成的号码满足配置的约束条件",
            "",
            "**统计说明：** Z分数 > |1.0| 表示相对基线有统计显著性偏差。",
            "",
            "---",
            "*由超级大乐透统计分析框架生成*"
        ])
        
        # 保存报告
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(
        description='超级大乐透统计分析与候选号码生成框架',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""使用示例：
  python scripts/run_analysis.py --config config.yaml
  python scripts/run_analysis.py --config config.yaml --output results
        
更多信息请参考 README.md""")
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径 (default: config.yaml)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出目录 (覆盖config.yaml中的设置)')
    
    args = parser.parse_args()
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found.")
        print("Please create a config.yaml file or specify the correct path.")
        sys.exit(1)
    
    try:
        # 初始化分析框架
        print(f"Initializing lottery analysis framework...")
        framework = LotteryAnalysisFramework(args.config)
        
        # 覆盖输出目录设置
        if args.output:
            framework.config['output']['output_dir'] = args.output
        
        # 加载数据
        framework.load_data()
        
        # 运行分析
        analysis_results = framework.run_analysis()
        
        # 生成候选号码
        candidates = framework.generate_candidates(analysis_results)
        
        # 保存结果
        framework.save_results(analysis_results, candidates)
        
        # 打印摘要
        print("\n" + "="*60)
        print("分析完成摘要")
        print("="*60)
        print(f"检测偏向: {analysis_results['bias'].upper()}")
        print(f"数据集: {analysis_results['total_draws']} 期开奖")
        print(f"趋势窗口: {analysis_results['recent_window']} 期")
        print(f"生成候选: {len(candidates)} 注")
        print("\n生成的候选号码:")
        
        for candidate in candidates:
            front_str = ' '.join([f"{num:02d}" for num in candidate['front_numbers']])
            back_str = ' '.join([f"{num:02d}" for num in candidate['back_numbers']])
            print(f"  {candidate['id']}. {front_str} + {back_str}")
        
        print("\n⚠️ 仅供娱乐参考 - 不保证任何中奖概率提升！")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()