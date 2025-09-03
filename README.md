# 超级大乐透 Lottery Analysis Toolchain

**⚠️ ENTERTAINMENT PURPOSES ONLY ⚠️**

This toolchain analyzes historical lottery data for educational and entertainment purposes only. All lottery number combinations are theoretically equiprobable in a fair lottery. This tool does NOT guarantee any improved probability of winning.

## Features

- **Statistical Analysis**: Analyzes 500+ historical draws with frequency calculations
- **Trend Detection**: Computes z-scores to identify hot/cold number trends
- **Bias Classification**: Automatically detects hot/cold/neutral bias patterns
- **Pick Generation**: Creates 5 candidate picks using weighted randomization
- **GitHub Actions Integration**: Automated analysis on commits and manual triggers

## Usage

### Manual Analysis

```bash
# Run analysis with default settings (50 draw recent window)
python3 lottery_analysis.py

# Custom recent window size
python3 lottery_analysis.py --recent 30

# Custom file paths
python3 lottery_analysis.py --input data.txt --output report.md --picks picks.txt
```

### GitHub Actions

The analysis runs automatically on:
- Push to main/master branch
- Pull requests to main/master branch
- Manual workflow dispatch (with configurable recent window)

To run manually:
1. Go to Actions tab in GitHub
2. Select "Lottery Analysis" workflow
3. Click "Run workflow"
4. Optionally set custom recent window size

## Files

- `lottery_analysis.py` - Main analysis script
- `analysis_report.md` - Generated statistical analysis report
- `candidate_picks.txt` - Generated lottery picks
- `.github/workflows/lottery-analysis.yml` - GitHub Actions workflow

## Methodology

1. **Data Loading**: Parses JSON file with historical draw results
2. **Baseline Analysis**: Calculates frequency statistics across all draws
3. **Recent Window Analysis**: Analyzes trends in recent N draws
4. **Z-Score Calculation**: Measures statistical significance of trends
5. **Bias Detection**: Classifies overall trend as hot/cold/neutral
6. **Pick Generation**: Creates weighted random picks based on detected bias

## Statistical Notes

- Z-scores > |1.0| indicate statistically significant deviations
- Neutral bias: |mean_delta| < 0.002 and < 55% of numbers have |z| >= 1
- Hot bias: Majority positive z-scores with mean_delta >= +0.002  
- Cold bias: Majority negative z-scores with mean_delta <= -0.002

## Disclaimer

This analysis is provided for educational and entertainment purposes only. Lottery games involve significant financial risk. All number combinations in a fair lottery are theoretically equiprobable regardless of historical patterns. Play responsibly and never spend more than you can afford to lose.