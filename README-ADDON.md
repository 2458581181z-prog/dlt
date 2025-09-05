# Wheel Lottery Add-on

## How to use
1. Merge this branch into `main`.
2. Go to Actions → "Wheel Lottery Picks".
3. Click "Run workflow", set "tickets" if needed (default 5), then run.
4. The workflow will:
   - run `scripts/wheel_picks.py`
   - write files into `results/`
   - automatically open a PR with the new results.

You can change numbers per ticket with `--numbers` and range with `--max` directly in the script if desired.
