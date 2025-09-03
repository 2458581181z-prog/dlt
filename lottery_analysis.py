#!/usr/bin/env python3
"""
Lottery Analysis Toolchain for 超级大乐透 (Super Lotto)

ENTERTAINMENT PURPOSES ONLY - This analysis is for educational and entertainment 
purposes only. All number combinations are theoretically equiprobable in a fair 
lottery. This tool does NOT guarantee any improved probability of winning.

Analysis includes:
- Historical frequency analysis
- Hot/cold number trend detection  
- Statistical z-score calculations
- Automated bias inference (hot/cold/neutral)
- Random candidate pick generation
"""

import json
import math
import statistics
import random
import collections
import datetime
import pathlib
import re
import argparse
import sys
from typing import List, Dict, Tuple, Any


class LotteryAnalysis:
    """Main lottery analysis class."""
    
    def __init__(self, recent_window: int = 50):
        self.recent_window = recent_window
        self.total_draws = 500
        self.front_range = (1, 35)  # Front numbers 1-35
        self.back_range = (1, 12)   # Back numbers 1-12
        self.draws_data = []
        
    def load_data(self, file_path: str) -> None:
        """Load and parse lottery draw data from JSON file."""
        print(f"Loading data from {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse multiple JSON objects in the file
        json_objects = []
        current_obj = ''
        brace_count = 0
        
        for line in content.split('\n'):
            if line.strip():
                current_obj += line + '\n'
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0 and current_obj.strip():
                    try:
                        obj = json.loads(current_obj.strip())
                        json_objects.append(obj)
                        current_obj = ''
                    except json.JSONDecodeError:
                        pass
        
        # Collect all draws from all JSON objects
        all_draws = []
        for obj in json_objects:
            if 'value' in obj and 'list' in obj['value']:
                all_draws.extend(obj['value']['list'])
        
        print(f"Found {len(all_draws)} total draws")
        
        # Use most recent 500 draws (or all if fewer)
        self.draws_data = all_draws[:self.total_draws]
        
        # Reverse to get chronological order (oldest -> newest)
        self.draws_data.reverse()
        
        print(f"Using {len(self.draws_data)} draws for analysis")
        print(f"Date range: {self.draws_data[0]['lotteryDrawTime']} to {self.draws_data[-1]['lotteryDrawTime']}")
    
    def parse_draw_result(self, draw_result: str) -> Tuple[List[int], List[int]]:
        """Parse draw result string into front and back numbers."""
        numbers = [int(x) for x in draw_result.split()]
        front_numbers = numbers[:5]
        back_numbers = numbers[5:]
        return front_numbers, back_numbers
    
    def calculate_frequencies(self, draws: List[Dict[str, Any]]) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Calculate frequency counts for front and back numbers."""
        front_freq = collections.defaultdict(int)
        back_freq = collections.defaultdict(int)
        
        for draw in draws:
            front_nums, back_nums = self.parse_draw_result(draw['lotteryDrawResult'])
            
            for num in front_nums:
                front_freq[num] += 1
            for num in back_nums:
                back_freq[num] += 1
        
        return dict(front_freq), dict(back_freq)
    
    def calculate_z_scores(self, baseline_freq: Dict[int, int], recent_freq: Dict[int, int], 
                          total_baseline: int, total_recent: int, num_range: Tuple[int, int]) -> Dict[int, float]:
        """Calculate z-scores for trend analysis."""
        z_scores = {}
        
        for num in range(num_range[0], num_range[1] + 1):
            baseline_count = baseline_freq.get(num, 0)
            recent_count = recent_freq.get(num, 0)
            
            # Calculate probabilities
            p_base = baseline_count / total_baseline if total_baseline > 0 else 0
            p_recent = recent_count / total_recent if total_recent > 0 else 0
            
            # Calculate z-score with safe denominator clipping
            if p_base == 0 or p_base == 1:
                # Use small epsilon for extreme cases
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
    
    def determine_bias(self, front_z_scores: Dict[int, float], back_z_scores: Dict[int, float],
                      front_baseline_freq: Dict[int, int], front_recent_freq: Dict[int, int],
                      back_baseline_freq: Dict[int, int], back_recent_freq: Dict[int, int],
                      total_baseline: int, total_recent: int) -> str:
        """Determine overall bias (hot/cold/neutral) based on statistical analysis."""
        
        # Calculate deltas for all numbers
        deltas = []
        z_values = []
        
        # Front numbers
        for num in range(self.front_range[0], self.front_range[1] + 1):
            baseline_prob = front_baseline_freq.get(num, 0) / total_baseline
            recent_prob = front_recent_freq.get(num, 0) / total_recent
            delta = recent_prob - baseline_prob
            deltas.append(delta)
            z_values.append(front_z_scores[num])
        
        # Back numbers  
        for num in range(self.back_range[0], self.back_range[1] + 1):
            baseline_prob = back_baseline_freq.get(num, 0) / total_baseline
            recent_prob = back_recent_freq.get(num, 0) / total_recent
            delta = recent_prob - baseline_prob
            deltas.append(delta)
            z_values.append(back_z_scores[num])
        
        # Calculate statistics
        mean_delta = statistics.mean(deltas) if deltas else 0
        high_z_count = sum(1 for z in z_values if abs(z) >= 1.0)
        high_z_ratio = high_z_count / len(z_values) if z_values else 0
        
        positive_z_count = sum(1 for z in z_values if z > 0)
        negative_z_count = sum(1 for z in z_values if z < 0)
        
        # Determine bias
        if abs(mean_delta) < 0.002 and high_z_ratio < 0.55:
            return "neutral"
        elif positive_z_count > negative_z_count and mean_delta >= 0.002:
            return "hot"
        elif negative_z_count > positive_z_count and mean_delta <= -0.002:
            return "cold"
        else:
            return "neutral"
    
    def generate_picks(self, bias: str, front_z_scores: Dict[int, float], 
                      back_z_scores: Dict[int, float], num_picks: int = 5) -> List[Tuple[List[int], List[int]]]:
        """Generate candidate lottery picks based on bias analysis."""
        picks = []
        
        for _ in range(num_picks):
            # Generate front numbers (5 numbers from 1-35)
            front_candidates = list(range(self.front_range[0], self.front_range[1] + 1))
            
            if bias == "hot":
                # Favor numbers with positive z-scores, but still include randomness
                weights = [max(0.1, 1 + front_z_scores.get(num, 0) * 0.3) for num in front_candidates]
            elif bias == "cold":
                # Favor numbers with negative z-scores
                weights = [max(0.1, 1 - front_z_scores.get(num, 0) * 0.3) for num in front_candidates]
            else:
                # Neutral - equal weights
                weights = [1.0] * len(front_candidates)
            
            front_pick = random.choices(front_candidates, weights=weights, k=5)
            front_pick = sorted(list(set(front_pick)))
            
            # Ensure we have exactly 5 unique numbers
            while len(front_pick) < 5:
                remaining = [n for n in front_candidates if n not in front_pick]
                if remaining:
                    if bias == "hot":
                        weights_remaining = [max(0.1, 1 + front_z_scores.get(num, 0) * 0.3) for num in remaining]
                    elif bias == "cold":
                        weights_remaining = [max(0.1, 1 - front_z_scores.get(num, 0) * 0.3) for num in remaining]
                    else:
                        weights_remaining = [1.0] * len(remaining)
                    additional = random.choices(remaining, weights=weights_remaining, k=1)
                    front_pick.extend(additional)
                    front_pick = sorted(list(set(front_pick)))
            
            front_pick = sorted(front_pick[:5])
            
            # Generate back numbers (2 numbers from 1-12)
            back_candidates = list(range(self.back_range[0], self.back_range[1] + 1))
            
            if bias == "hot":
                back_weights = [max(0.1, 1 + back_z_scores.get(num, 0) * 0.3) for num in back_candidates]
            elif bias == "cold":
                back_weights = [max(0.1, 1 - back_z_scores.get(num, 0) * 0.3) for num in back_candidates]
            else:
                back_weights = [1.0] * len(back_candidates)
            
            back_pick = random.choices(back_candidates, weights=back_weights, k=2)
            back_pick = sorted(list(set(back_pick)))
            
            # Ensure we have exactly 2 unique numbers
            while len(back_pick) < 2:
                remaining = [n for n in back_candidates if n not in back_pick]
                if remaining:
                    if bias == "hot":
                        weights_remaining = [max(0.1, 1 + back_z_scores.get(num, 0) * 0.3) for num in remaining]
                    elif bias == "cold":
                        weights_remaining = [max(0.1, 1 - back_z_scores.get(num, 0) * 0.3) for num in remaining]
                    else:
                        weights_remaining = [1.0] * len(remaining)
                    additional = random.choices(remaining, weights=weights_remaining, k=1)
                    back_pick.extend(additional)
                    back_pick = sorted(list(set(back_pick)))
            
            back_pick = sorted(back_pick[:2])
            picks.append((front_pick, back_pick))
        
        return picks
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run complete lottery analysis."""
        print("Starting lottery analysis...")
        
        # Calculate baseline frequencies (all draws)
        baseline_front_freq, baseline_back_freq = self.calculate_frequencies(self.draws_data)
        total_baseline_draws = len(self.draws_data)
        
        # Calculate recent frequencies (last N draws)
        recent_draws = self.draws_data[-self.recent_window:]
        recent_front_freq, recent_back_freq = self.calculate_frequencies(recent_draws)
        total_recent_draws = len(recent_draws)
        
        print(f"Baseline analysis: {total_baseline_draws} draws")
        print(f"Recent analysis: {total_recent_draws} draws")
        
        # Calculate z-scores
        front_z_scores = self.calculate_z_scores(
            baseline_front_freq, recent_front_freq, 
            total_baseline_draws, total_recent_draws, self.front_range
        )
        
        back_z_scores = self.calculate_z_scores(
            baseline_back_freq, recent_back_freq,
            total_baseline_draws, total_recent_draws, self.back_range
        )
        
        # Determine bias
        bias = self.determine_bias(
            front_z_scores, back_z_scores,
            baseline_front_freq, recent_front_freq,
            baseline_back_freq, recent_back_freq,
            total_baseline_draws, total_recent_draws
        )
        
        print(f"Detected bias: {bias}")
        
        # Generate picks
        picks = self.generate_picks(bias, front_z_scores, back_z_scores)
        
        return {
            'bias': bias,
            'baseline_draws': total_baseline_draws,
            'recent_draws': total_recent_draws,
            'recent_window': self.recent_window,
            'front_z_scores': front_z_scores,
            'back_z_scores': back_z_scores,
            'baseline_front_freq': baseline_front_freq,
            'recent_front_freq': recent_front_freq,
            'baseline_back_freq': baseline_back_freq,
            'recent_back_freq': recent_back_freq,
            'picks': picks,
            'analysis_date': datetime.datetime.now().isoformat()
        }
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate analysis report."""
        report = []
        report.append("# 超级大乐透 Lottery Analysis Report")
        report.append("")
        report.append("**⚠️ ENTERTAINMENT PURPOSES ONLY ⚠️**")
        report.append("")
        report.append("This analysis is provided for educational and entertainment purposes only.")
        report.append("All lottery number combinations are theoretically equiprobable in a fair lottery.")
        report.append("This tool does NOT guarantee any improved probability of winning.")
        report.append("Lottery games involve significant financial risk - play responsibly.")
        report.append("")
        
        report.append(f"**Analysis Date:** {results['analysis_date']}")
        report.append(f"**Dataset:** {results['baseline_draws']} historical draws")
        report.append(f"**Recent Window:** {results['recent_draws']} draws")
        report.append(f"**Detected Bias:** {results['bias'].upper()}")
        report.append("")
        
        # Statistical Summary
        report.append("## Statistical Summary")
        report.append("")
        
        # Front numbers analysis
        report.append("### Front Numbers (1-35)")
        front_z_sorted = sorted(results['front_z_scores'].items(), key=lambda x: x[1], reverse=True)
        
        report.append("**Top 10 Hot Numbers (Positive Z-scores):**")
        hot_front = [(num, z) for num, z in front_z_sorted if z > 0][:10]
        for num, z in hot_front:
            baseline_freq = results['baseline_front_freq'].get(num, 0)
            recent_freq = results['recent_front_freq'].get(num, 0)
            report.append(f"- {num:2d}: z={z:+.2f} (baseline: {baseline_freq}, recent: {recent_freq})")
        
        report.append("")
        report.append("**Top 10 Cold Numbers (Negative Z-scores):**")
        cold_front = [(num, z) for num, z in front_z_sorted if z <= 0][-10:]
        cold_front.reverse()
        for num, z in cold_front:
            baseline_freq = results['baseline_front_freq'].get(num, 0)
            recent_freq = results['recent_front_freq'].get(num, 0)
            report.append(f"- {num:2d}: z={z:+.2f} (baseline: {baseline_freq}, recent: {recent_freq})")
        
        # Back numbers analysis
        report.append("")
        report.append("### Back Numbers (1-12)")
        back_z_sorted = sorted(results['back_z_scores'].items(), key=lambda x: x[1], reverse=True)
        
        report.append("**Hot Numbers (Positive Z-scores):**")
        hot_back = [(num, z) for num, z in back_z_sorted if z > 0]
        for num, z in hot_back:
            baseline_freq = results['baseline_back_freq'].get(num, 0)
            recent_freq = results['recent_back_freq'].get(num, 0)
            report.append(f"- {num:2d}: z={z:+.2f} (baseline: {baseline_freq}, recent: {recent_freq})")
        
        report.append("")
        report.append("**Cold Numbers (Negative Z-scores):**")
        cold_back = [(num, z) for num, z in back_z_sorted if z <= 0]
        cold_back.reverse()
        for num, z in cold_back:
            baseline_freq = results['baseline_back_freq'].get(num, 0)
            recent_freq = results['recent_back_freq'].get(num, 0)
            report.append(f"- {num:2d}: z={z:+.2f} (baseline: {baseline_freq}, recent: {recent_freq})")
        
        # Generated picks
        report.append("")
        report.append("## Generated Candidate Picks")
        report.append("")
        report.append(f"Based on {results['bias']} bias analysis:")
        report.append("")
        
        for i, (front, back) in enumerate(results['picks'], 1):
            front_str = ' '.join(f"{n:02d}" for n in front)
            back_str = ' '.join(f"{n:02d}" for n in back)
            report.append(f"**Pick {i}:** {front_str} + {back_str}")
        
        report.append("")
        report.append("## Methodology")
        report.append("")
        report.append("1. **Baseline Analysis:** Frequency calculation across all historical draws")
        report.append("2. **Recent Window:** Frequency calculation for the most recent draws")
        report.append("3. **Z-Score Calculation:** Statistical significance of recent vs baseline trends")
        report.append("4. **Bias Detection:** Automated classification as hot/cold/neutral based on statistical thresholds")
        report.append("5. **Pick Generation:** Weighted random selection incorporating detected bias")
        report.append("")
        report.append("**Statistical Note:** Z-scores > |1.0| indicate statistically significant deviations from baseline.")
        report.append("")
        report.append("---")
        report.append("*Generated by DLT Lottery Analysis Toolchain*")
        
        return '\n'.join(report)
    
    def save_picks(self, results: Dict[str, Any], filename: str = 'candidate_picks.txt') -> None:
        """Save candidate picks to a text file."""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("超级大乐透 Candidate Picks\n")
            f.write("=" * 30 + "\n")
            f.write(f"Generated: {results['analysis_date']}\n")
            f.write(f"Bias: {results['bias'].upper()}\n")
            f.write("\n⚠️ ENTERTAINMENT PURPOSES ONLY ⚠️\n")
            f.write("This tool does NOT guarantee improved winning probability.\n")
            f.write("All combinations are theoretically equiprobable.\n")
            f.write("\n")
            
            for i, (front, back) in enumerate(results['picks'], 1):
                front_str = ' '.join(f"{n:02d}" for n in front)
                back_str = ' '.join(f"{n:02d}" for n in back)
                f.write(f"Pick {i}: {front_str} + {back_str}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Lottery Analysis Toolchain for 超级大乐透')
    parser.add_argument('--recent', type=int, default=50, 
                       help='Size of recent window for trend analysis (default: 50)')
    parser.add_argument('--input', type=str, default='新建文本文档 (6).txt',
                       help='Input JSON file path (default: 新建文本文档 (6).txt)')
    parser.add_argument('--output', type=str, default='analysis_report.md',
                       help='Output report file path (default: analysis_report.md)')
    parser.add_argument('--picks', type=str, default='candidate_picks.txt',
                       help='Output picks file path (default: candidate_picks.txt)')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not pathlib.Path(args.input).exists():
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
    
    try:
        # Initialize analysis
        analyzer = LotteryAnalysis(recent_window=args.recent)
        
        # Load data and run analysis
        analyzer.load_data(args.input)
        results = analyzer.run_analysis()
        
        # Generate and save report
        report = analyzer.generate_report(results)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Analysis report saved to: {args.output}")
        
        # Save picks
        analyzer.save_picks(results, args.picks)
        print(f"Candidate picks saved to: {args.picks}")
        
        # Print summary
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        print(f"Detected Bias: {results['bias'].upper()}")
        print(f"Dataset: {results['baseline_draws']} draws")
        print(f"Recent Window: {results['recent_draws']} draws")
        print("\nGenerated Picks:")
        for i, (front, back) in enumerate(results['picks'], 1):
            front_str = ' '.join(f"{n:02d}" for n in front)
            back_str = ' '.join(f"{n:02d}" for n in back)
            print(f"  {i}. {front_str} + {back_str}")
        print("\n⚠️ ENTERTAINMENT PURPOSES ONLY - No guaranteed probability improvement!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()