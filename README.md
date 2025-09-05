# 超级大乐透统计分析框架 / Super Lotto Statistical Analysis Framework

**⚠️ 仅供娱乐参考 ⚠️ / ENTERTAINMENT PURPOSES ONLY ⚠️**

基于历史开奖数据的统计分析与候选号码生成框架。本工具仅供教育和娱乐目的，所有号码组合理论上等概率，不保证任何中奖概率提升。

A statistical analysis and candidate number generation framework based on historical lottery data. This tool is for educational and entertainment purposes only. All number combinations are theoretically equiprobable, and no improved winning probability is guaranteed.

## 快速开始 / Quick Start

### 1. 安装依赖 / Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. 准备数据 / Prepare Data

将历史开奖数据放置到 `data/history.csv`（CSV格式）或配置现有的JSON数据文件：

Place historical lottery data in `data/history.csv` (CSV format) or configure existing JSON data file:

**CSV格式示例 / CSV Format Example:**
```csv
draw_date,draw_num,front_numbers,back_numbers
2025-09-01,25100,"26,28,32,34,35","02,07"
2025-08-29,25099,"03,06,14,19,25","06,11"
```

**或使用现有JSON格式 / Or use existing JSON format** (compatible with current data)

### 3. 运行分析 / Run Analysis

```bash
# 使用默认配置 / Use default config
python scripts/run_analysis.py --config config.yaml

# 自定义输出目录 / Custom output directory  
python scripts/run_analysis.py --config config.yaml --output results
```

### 4. 查看结果 / View Results

生成的文件 / Generated files:
- `candidates_YYYYMMDD_HHMMSS.csv` - 候选号码及详细理由 / Candidate numbers with detailed reasoning
- `report_YYYYMMDD_HHMMSS.md` - 完整分析报告 / Complete analysis report

## 数据格式 / Data Format

### CSV格式 / CSV Format

配置文件中可自定义列名映射 / Column name mapping can be customized in config file:

```yaml
data:
  columns:
    date: "draw_date"           # 开奖日期列
    draw_number: "draw_num"     # 期号列  
    front_numbers: "front_numbers" # 前区号码列
    back_numbers: "back_numbers"   # 后区号码列
```

### JSON格式 / JSON Format

兼容现有的彩票数据JSON格式 / Compatible with existing lottery data JSON format.

## 方法概述 / Methodology Overview

### 统计指标 / Statistical Metrics

1. **冷热号分析 / Hot/Cold Number Analysis** - 基于频率和Z分数的趋势分析
2. **遗漏值计算 / Missing Value Calculation** - 号码未出现的期数统计  
3. **和值分析 / Sum Value Analysis** - 号码和值的分布统计
4. **奇偶比分析 / Odd/Even Ratio Analysis** - 奇偶数字的比例分析
5. **区间分布 / Range Distribution** - 号码在不同区间的分布
6. **连号分析 / Consecutive Numbers** - 连续号码的出现统计
7. **同尾分析 / Same Tail Analysis** - 相同尾数号码的统计
8. **关联度分析 / Correlation Analysis** - 号码间的相关性分析

### 生成策略 / Generation Strategy

1. **权重计算 / Weight Calculation** - 综合多种指标计算号码选择权重
2. **约束验证 / Constraint Validation** - 确保生成号码满足配置约束
3. **理由生成 / Reasoning Generation** - 为每注号码生成详细选择理由

### 配置参数 / Configuration Parameters

通过 `config.yaml` 可配置：
- 数据路径和列映射 / Data path and column mapping
- 分析参数（窗口大小、号码范围等）/ Analysis parameters  
- 指标权重 / Metrics weights
- 约束条件 / Constraints
- 输出格式 / Output format

## 文件结构 / File Structure

```
├── config.yaml              # 配置文件 / Configuration file
├── requirements.txt          # 依赖列表 / Dependencies  
├── scripts/
│   └── run_analysis.py      # 主程序 / Main script
├── src/
│   ├── metrics.py           # 统计指标模块 / Metrics module
│   └── model.py             # 生成模型模块 / Model module  
├── data/
│   └── history.csv          # 数据文件 / Data file
└── output/                  # 输出目录 / Output directory
    ├── candidates_*.csv     # 候选号码 / Candidate numbers
    └── report_*.md          # 分析报告 / Analysis reports
```

## 使用示例 / Usage Examples

### 基本使用 / Basic Usage
```bash
python scripts/run_analysis.py --config config.yaml
```

### 自定义输出 / Custom Output
```bash
python scripts/run_analysis.py --config config.yaml --output my_results
```

### 查看帮助 / View Help
```bash
python scripts/run_analysis.py --help
```

## 配置说明 / Configuration Guide

主要配置项 / Key configuration options:

- `data.input_file` - 数据文件路径 / Data file path
- `analysis.recent_window` - 趋势分析窗口大小 / Trend analysis window size  
- `metrics.*_weight` - 各指标权重 / Individual metrics weights
- `constraints.*` - 约束条件 / Constraint conditions
- `output.num_candidates` - 生成候选数量 / Number of candidates to generate

## 免责声明 / Disclaimer

本分析框架仅供教育和娱乐目的。彩票游戏涉及重大财务风险。在公平的彩票中，所有号码组合理论上等概率，不受历史模式影响。请理性投注，量力而行。

This analysis framework is provided for educational and entertainment purposes only. Lottery games involve significant financial risk. All number combinations in a fair lottery are theoretically equiprobable regardless of historical patterns. Play responsibly and never spend more than you can afford to lose.

## 扩展功能 / Future Extensions

- 更复杂的关联度分析（互信息/phi系数）/ Advanced correlation analysis
- 蒙特卡洛重排验证 / Monte Carlo permutation validation  
- 历史回测与命中统计 / Historical backtesting and hit statistics
- GitHub Actions 定时任务 / Scheduled GitHub Actions tasks