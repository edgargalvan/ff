"""
DFS lineup optimizer using integer linear programming.
Replaces the MATLAB/CPLEX naive_picker.m with Python/PuLP.
"""

import pulp
import polars as pl


# Default constraints
DEFAULT_SALARY_CAP = 60000
DEFAULT_ROSTER_SIZE = 9
DEFAULT_POSITION_LIMITS = {
    "QB": 1,
    "WR": 3,
    "RB": 2,
    "TE": 1,
    "K": 1,
    "DEF": 1,
}
DEFAULT_MAX_PER_TEAM = 4


def optimize_lineup(
    players,
    salary_cap=DEFAULT_SALARY_CAP,
    roster_size=DEFAULT_ROSTER_SIZE,
    position_limits=None,
    max_per_team=DEFAULT_MAX_PER_TEAM,
    exclude=None,
):
    """
    Find the optimal DFS lineup using integer linear programming.

    Args:
        players: polars DataFrame with columns:
            - name: player name
            - position: QB, WR, RB, TE, K, or DEF
            - team: team abbreviation
            - salary: player cost
            - projected_points: expected fantasy points
        salary_cap: maximum total salary
        roster_size: number of players on roster
        position_limits: dict of position -> max count (defaults to standard DFS)
        max_per_team: max players from any single team
        exclude: list of player names to exclude

    Returns:
        polars DataFrame of selected players, or None if infeasible
    """
    if position_limits is None:
        position_limits = DEFAULT_POSITION_LIMITS

    if exclude is None:
        exclude = []

    # Filter out excluded players
    pool = players.filter(~pl.col("name").is_in(exclude))
    n = len(pool)

    if n == 0:
        return None

    # Extract data as lists
    names = pool["name"].to_list()
    positions = pool["position"].to_list()
    teams = pool["team"].to_list()
    salaries = pool["salary"].to_list()
    points = pool["projected_points"].to_list()

    # Create the optimization problem
    prob = pulp.LpProblem("DFS_Lineup", pulp.LpMaximize)

    # Binary decision variables: 1 if player is selected, 0 otherwise
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    # Objective: maximize total projected points
    prob += pulp.lpSum(points[i] * x[i] for i in range(n))

    # Constraint: salary cap
    prob += pulp.lpSum(salaries[i] * x[i] for i in range(n)) <= salary_cap

    # Constraint: roster size
    prob += pulp.lpSum(x[i] for i in range(n)) == roster_size

    # Constraint: position limits
    for pos, limit in position_limits.items():
        indices = [i for i in range(n) if positions[i] == pos]
        if indices:
            prob += pulp.lpSum(x[i] for i in indices) <= limit

    # Constraint: max players per team
    unique_teams = set(teams)
    for team in unique_teams:
        indices = [i for i in range(n) if teams[i] == team]
        if indices:
            prob += pulp.lpSum(x[i] for i in indices) <= max_per_team

    # Solve
    solver = pulp.COIN_CMD(msg=0) if pulp.COIN_CMD().available() else pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)

    if prob.status != pulp.constants.LpStatusOptimal:
        print("No optimal solution found.")
        return None

    # Extract selected players
    selected = []
    for i in range(n):
        if x[i].varValue and x[i].varValue > 0.5:
            selected.append({
                "name": names[i],
                "position": positions[i],
                "team": teams[i],
                "salary": salaries[i],
                "projected_points": points[i],
            })

    result = pl.DataFrame(selected)
    return result


def find_multiple_lineups(players, num_lineups=5, **kwargs):
    """
    Generate multiple distinct optimal lineups by excluding previous solutions.

    Each subsequent lineup is the best lineup that doesn't exactly match
    any previously found lineup.
    """
    lineups = []
    exclude_constraints = []

    pool = players.filter(~pl.col("name").is_in(kwargs.get("exclude", [])))
    n = len(pool)
    names = pool["name"].to_list()
    positions = pool["position"].to_list()
    teams = pool["team"].to_list()
    salaries = pool["salary"].to_list()
    points = pool["projected_points"].to_list()

    salary_cap = kwargs.get("salary_cap", DEFAULT_SALARY_CAP)
    roster_size = kwargs.get("roster_size", DEFAULT_ROSTER_SIZE)
    position_limits = kwargs.get("position_limits", DEFAULT_POSITION_LIMITS)
    max_per_team = kwargs.get("max_per_team", DEFAULT_MAX_PER_TEAM)

    for k in range(num_lineups):
        prob = pulp.LpProblem(f"DFS_Lineup_{k}", pulp.LpMaximize)
        x = [pulp.LpVariable(f"x_{k}_{i}", cat="Binary") for i in range(n)]

        prob += pulp.lpSum(points[i] * x[i] for i in range(n))
        prob += pulp.lpSum(salaries[i] * x[i] for i in range(n)) <= salary_cap
        prob += pulp.lpSum(x[i] for i in range(n)) == roster_size

        for pos, limit in position_limits.items():
            indices = [i for i in range(n) if positions[i] == pos]
            if indices:
                prob += pulp.lpSum(x[i] for i in indices) <= limit

        unique_teams = set(teams)
        for team in unique_teams:
            indices = [i for i in range(n) if teams[i] == team]
            if indices:
                prob += pulp.lpSum(x[i] for i in indices) <= max_per_team

        # Exclude previous lineups: the new lineup can't have the exact same set
        for prev_indices in exclude_constraints:
            prob += pulp.lpSum(x[i] for i in prev_indices) <= len(prev_indices) - 1

        solver = pulp.COIN_CMD(msg=0) if pulp.COIN_CMD().available() else pulp.PULP_CBC_CMD(msg=0)
        prob.solve(solver)

        if prob.status != pulp.constants.LpStatusOptimal:
            break

        selected_indices = [i for i in range(n) if x[i].varValue and x[i].varValue > 0.5]
        exclude_constraints.append(selected_indices)

        lineup = []
        for i in selected_indices:
            lineup.append({
                "name": names[i],
                "position": positions[i],
                "team": teams[i],
                "salary": salaries[i],
                "projected_points": points[i],
            })

        lineups.append(pl.DataFrame(lineup))

    return lineups


if __name__ == "__main__":
    # Demo with sample data
    sample_players = pl.DataFrame({
        "name": [
            "Josh Allen", "Jalen Hurts", "Lamar Jackson",
            "Derrick Henry", "Saquon Barkley", "Josh Jacobs", "Bijan Robinson",
            "Ja'Marr Chase", "Amon-Ra St. Brown", "CeeDee Lamb", "Tyreek Hill", "Davante Adams",
            "Travis Kelce", "Sam LaPorta",
            "Tyler Bass", "Jake Elliott",
            "BUF DEF", "PHI DEF",
        ],
        "position": [
            "QB", "QB", "QB",
            "RB", "RB", "RB", "RB",
            "WR", "WR", "WR", "WR", "WR",
            "TE", "TE",
            "K", "K",
            "DEF", "DEF",
        ],
        "team": [
            "BUF", "PHI", "BAL",
            "BAL", "PHI", "GB", "ATL",
            "CIN", "DET", "DAL", "MIA", "LV",
            "KC", "DET",
            "BUF", "PHI",
            "BUF", "PHI",
        ],
        "salary": [
            8500, 8200, 8400,
            7200, 7800, 6500, 7500,
            8000, 7200, 7600, 7400, 6800,
            6500, 5800,
            4500, 4200,
            4000, 4200,
        ],
        "projected_points": [
            24.5, 22.1, 23.8,
            16.2, 18.5, 14.8, 17.1,
            19.5, 17.8, 18.2, 16.5, 15.2,
            12.5, 11.2,
            8.5, 7.8,
            9.0, 8.5,
        ],
    })

    print("Finding optimal lineup...")
    print("=" * 60)

    lineup = optimize_lineup(sample_players)
    if lineup is not None:
        print(lineup)
        total_pts = lineup["projected_points"].sum()
        total_sal = lineup["salary"].sum()
        print(f"\nTotal projected: {total_pts:.1f} pts")
        print(f"Total salary:    ${total_sal:,}")

    print("\n\nFinding top 3 lineups...")
    print("=" * 60)

    lineups = find_multiple_lineups(sample_players, num_lineups=3)
    for i, lu in enumerate(lineups):
        print(f"\nLineup #{i + 1}:")
        print(lu)
        print(f"  Total: {lu['projected_points'].sum():.1f} pts, ${lu['salary'].sum():,}")
