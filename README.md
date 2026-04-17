# ff

NFL game prediction via hierarchical Bayesian modeling, with fantasy scoring and DFS lineup optimization.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Project Structure

```
src/
  config.py        - Fantasy scoring rules
  data.py          - NFL game data loading via nflreadpy
  model.py         - Hierarchical Bayesian model (PyMC)
  stats.py         - Player/team stats, fantasy point calculations
  analysis.py      - Opponent-strength analysis
  optimizers/      - DFS lineup optimizer (PuLP/CBC)
notebooks/
  football.ipynb   - Model fitting, simulation, and spread prediction
tests/             - 110 tests
```

## Core Model

Estimates per-team attack and defense strengths using a Poisson log-linear model:

```
home_score ~ Poisson(exp(intercept + home + attack[home_team] + defense[away_team]))
away_score ~ Poisson(exp(intercept + attack[away_team] + defense[home_team]))
```

Fitted via MCMC (NUTS) with PyMC. Posterior draws are used to simulate games and generate spread predictions with credible intervals.

```python
from src.data import load_game_data
from src.model import bhm, simulate_team_seasons, predictions

df, teams = load_game_data(2024, weeks=list(range(1, 15)))
idata = bhm(df, metric="score", samples=1000)
```

## Tests

```
pytest
```
