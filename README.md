# ff

Hierarchical Bayesian model for predicting NFL game outcomes, with fantasy football scoring and DFS lineup optimization.

## Overview

The core of this project is a **Negative Binomial log-linear model** that estimates per-team offensive and defensive strengths from historical game data, then simulates future games to generate spread predictions with credible intervals.

```
home_score ~ NegBin(mu = exp(intercept + home + attack[home_team] + defense[away_team] + covariates), alpha)
away_score ~ NegBin(mu = exp(intercept + attack[away_team] + defense[home_team] + covariates), alpha)
```

Key features:
- **Negative Binomial likelihood** handles the overdispersion in NFL scores (scoring in 3s and 7s)
- **Non-centered parameterization** for efficient NUTS sampling with 32 teams
- **Time-varying team strengths** (optional) via GaussianRandomWalk
- **Game covariates** (optional): rest days, weather, indoor/outdoor, divisional rivalry
- **Rolling backtest** to validate predictions against actual outcomes

## Results

Rolling backtest on the 2024 NFL season (train on 8-week windows, predict next week):

| Metric | Value |
|---|---|
| Games predicted | 64 (weeks 15-18) |
| Winner accuracy | **68.8%** |
| Accuracy (confidence >= 60%) | **87.9%** (33 games) |
| Spread MAE | 11.2 points |
| Brier score | 0.205 (coin flip = 0.250) |

The model is well-calibrated: when it's more confident, it's more accurate. Games predicted with >= 60% confidence are correct ~88% of the time.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

### Fit and predict

```python
from src.data import load_game_data
from src.model import bhm
from src.backtest import predict_week

df, teams = load_game_data(2024, weeks=list(range(1, 15)))
idata = bhm(df, metric="score", samples=1000)

df_test, _ = load_game_data(2024, weeks=[15])
preds = predict_week(idata, df_test, nsims=500)
```

### Run a backtest

```python
from src.backtest import backtest, print_summary

results = backtest(2024, train_window=8, nsims=500, samples=500)
print_summary(results)
```

### Compare model variants

```python
from src.compare import compare_models, print_comparison

configs = [
    {"name": "base", "time_varying": False, "covariates": None},
    {"name": "+covariates", "covariates": ["rest_advantage", "temp_std"]},
    {"name": "+time_varying", "time_varying": True},
]
comparison = compare_models(2024, configs)
print_comparison(comparison)
```

## Project Structure

```
src/
  model.py                   Hierarchical Bayesian model (PyMC)
  data.py                    NFL game data loading (nflreadpy)
  backtest.py                Rolling backtest engine
  compare.py                 Model variant comparison
  config.py                  Fantasy scoring rules
  stats.py                   Player/team stats, fantasy point calculations
  analysis.py                Opponent-strength analysis
  optimizers/
    lineup_optimizer.py      DFS lineup optimizer (PuLP/CBC)
notebooks/
  football.ipynb             Model fitting, simulation, backtest
tests/                       131 tests
```

## Tests

```bash
pytest
```

## License

MIT
