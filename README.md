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
- **DFS lineup optimizer** using integer linear programming (PuLP)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

### Fit the model

```python
from src.data import load_game_data
from src.model import bhm, simulate_team_seasons, predictions

# Load 2024 regular season data (with covariates)
df, teams = load_game_data(2024, weeks=list(range(1, 15)))

# Fit base model
idata = bhm(df, metric="score", samples=1000)

# Or with covariates and time-varying strengths
idata = bhm(df, metric="score", samples=1000,
            time_varying=True,
            covariates=["rest_advantage", "temp_std", "wind_std"])
```

### Simulate and predict

```python
# Simulate test weeks
df_test, _ = load_game_data(2024, weeks=list(range(15, 19)))
simuls = simulate_team_seasons(df_test, idata, nsims=1000)
hdis = predictions(df_test, simuls, teams, nsims=1000)
```

### Fantasy scoring

```python
from src.stats import player_season_fantasy_points

weekly = player_season_fantasy_points("Josh Allen", 2024)
```

## Project Structure

```
src/
  config.py                  Fantasy scoring rules
  data.py                    NFL game data loading (nflreadpy)
  model.py                   Hierarchical Bayesian model (PyMC)
  stats.py                   Player/team stats, fantasy point calculations
  analysis.py                Opponent-strength analysis
  optimizers/
    lineup_optimizer.py      DFS lineup optimizer (PuLP/CBC)
notebooks/
  football.ipynb             Model fitting, simulation, spread prediction
tests/                       121 tests
```

## Tests

```bash
pytest
```

## License

MIT
