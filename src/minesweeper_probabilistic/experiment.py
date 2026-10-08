from __future__ import annotations

import csv
import os
from dataclasses import asdict
from typing import Callable

from .agent import GameStats, ProbabilisticAgent, RuleBasedAgent
from .board import MinesweeperBoard


CONFIGS = {
    "easy": (9, 9, 10),
    "medium": (16, 16, 40),
    "hard": (30, 16, 99),
}


def play_once(rows: int, cols: int, mines: int, agent_name: str, seed: int, utility: bool = False) -> GameStats:
    board = MinesweeperBoard(rows, cols, mines, seed=seed)
    if agent_name == "baseline":
        agent = RuleBasedAgent(seed=seed + 1)
    elif agent_name == "probabilistic":
        agent = ProbabilisticAgent(seed=seed + 1, exact_component_cap=64, mcmc_samples=1500, utility=utility)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")
    first_click = (rows // 2, cols // 2)
    return agent.play(board, first_click=first_click)


def run_experiments(
    games_per_config: int = 100,
    output_csv: str = "results/experiment_results.csv",
    seed_start: int = 20261008,
) -> list[dict]:
    rows: list[dict] = []
    for config_name, (r, c, m) in CONFIGS.items():
        for game_id in range(games_per_config):
            seed = seed_start + game_id
            for agent_name, utility in (("baseline", False), ("probabilistic", False), ("probabilistic", True)):
                stats = play_once(r, c, m, agent_name, seed, utility=utility)
                label = "probabilistic_safety" if agent_name == "probabilistic" and not utility else (
                    "probabilistic_utility" if utility else "baseline"
                )
                row = {
                    "configuration": config_name,
                    "rows": r,
                    "cols": c,
                    "mines": m,
                    "game_id": game_id,
                    "seed": seed,
                    "solver": label,
                    "won": int(stats.won),
                    "moves": stats.moves,
                    "elapsed_seconds": stats.elapsed_seconds,
                    "max_frontier": stats.max_frontier,
                    "exact_calls": stats.exact_calls,
                    "mcmc_calls": stats.mcmc_calls,
                    "fifty_fifty_clicks": stats.fifty_fifty_clicks,
                }
                rows.append(row)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarize_csv(input_csv: str) -> list[dict]:
    with open(input_csv, newline="", encoding="utf-8") as f:
        data = list(csv.DictReader(f))

    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in data:
        grouped.setdefault((row["configuration"], row["solver"]), []).append(row)

    summary = []
    for (configuration, solver), group in sorted(grouped.items()):
        n = len(group)
        summary.append(
            {
                "configuration": configuration,
                "solver": solver,
                "games": n,
                "wins": sum(int(x["won"]) for x in group),
                "win_rate": sum(int(x["won"]) for x in group) / n if n else 0.0,
                "avg_moves": sum(float(x["moves"]) for x in group) / n if n else 0.0,
                "avg_time_s": sum(float(x["elapsed_seconds"]) for x in group) / n if n else 0.0,
                "avg_max_frontier": sum(float(x["max_frontier"]) for x in group) / n if n else 0.0,
                "avg_exact_calls": sum(float(x["exact_calls"]) for x in group) / n if n else 0.0,
                "avg_mcmc_calls": sum(float(x["mcmc_calls"]) for x in group) / n if n else 0.0,
                "avg_50_50_clicks": sum(float(x["fifty_fifty_clicks"]) for x in group) / n if n else 0.0,
            }
        )
    return summary


if __name__ == "__main__":
    results = run_experiments(games_per_config=20)
    for row in summarize_csv("results/experiment_results.csv"):
        print(row)
