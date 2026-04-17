# ff

Fantasy football stats, scoring, and DFS lineup optimization.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Modules

- **config.py** - Fantasy scoring rules (points per stat, defense PA tiers)
- **stats.py** - Player/team stats via nflreadpy, fantasy point calculations
- **analysis.py** - Opponent-strength analysis (passing yards vs defensive matchups)
- **optimizers/lineup_optimizer.py** - DFS lineup optimizer (ILP via PuLP/CBC)

## Usage

```python
from stats import player_season_fantasy_points

# Get weekly fantasy points for a player
weekly = player_season_fantasy_points("Josh Allen", 2024)
print(weekly)
```

```python
from optimizers.lineup_optimizer import optimize_lineup
import polars as pl

# Optimize a DFS lineup from a player pool DataFrame
# (columns: name, position, team, salary, projected_points)
lineup = optimize_lineup(players)
```

## Tests

```
pytest
```
