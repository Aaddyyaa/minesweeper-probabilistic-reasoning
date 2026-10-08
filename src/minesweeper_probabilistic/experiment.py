from __future__ import annotations

import csv
import os
import time
from statistics import mean

from .agent import GameStats, ProbabilisticAgent, RuleBasedAgent
from .board import MinesweeperBoard


CONFIGS = {
    "easy": (9, 9, 10),
    "medium": (16, 16, 40),
    "hard": (30, 16, 99),
}


def play_once(
    rows: int, cols: int, mines: int, agent_name: str, seed: int, utility: bool = False
) -> GameStats:
    board = MinesweeperBoard(rows, cols, mines, seed=seed)
    if agent_name == "baseline":
        agent = RuleBasedAgent(seed=seed + 1)
    elif agent_name == "probabilistic":
        agent = ProbabilisticAgent(
            seed=seed + 1,
            exact_component_cap=28,
            exact_max_seconds=1.5,
            exact_max_nodes=1_000_000,
            mcmc_samples=3000,
            utility=utility,
        )
    else:
        raise ValueError(f"Unknown agent: {agent_name}")
    return agent.play(board, first_click=(rows // 2, cols // 2))


def run_experiments(
    games_per_config: int = 100,
    output_csv: str = "results/experiment_results.csv",
    seed_start: int = 20261008,
    progress_every: int = 25,
) -> list[dict]:
    if games_per_config <= 0:
        raise ValueError("games_per_config must be positive")

    rows: list[dict] = []
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    total = games_per_config * len(CONFIGS) * 3
    completed = 0
    overall_start = time.perf_counter()

    fieldnames = [
        "configuration","rows","cols","mines","game_id","seed","solver","won",
        "moves","elapsed_seconds","max_frontier","exact_calls","mcmc_calls",
        "heuristic_calls","fifty_fifty_clicks",
    ]

    for config_name, (r, c, m) in CONFIGS.items():
        for game_id in range(games_per_config):
            seed = seed_start + game_id
            for agent_name, utility in (
                ("baseline", False),
                ("probabilistic", False),
                ("probabilistic", True),
            ):
                stats = play_once(r, c, m, agent_name, seed, utility=utility)
                label = (
                    "probabilistic_utility" if utility else
                    "probabilistic_safety" if agent_name == "probabilistic"
                    else "baseline"
                )
                rows.append({
                    "configuration": config_name,
                    "rows": r, "cols": c, "mines": m,
                    "game_id": game_id, "seed": seed, "solver": label,
                    "won": int(stats.won), "moves": stats.moves,
                    "elapsed_seconds": stats.elapsed_seconds,
                    "max_frontier": stats.max_frontier,
                    "exact_calls": stats.exact_calls,
                    "mcmc_calls": stats.mcmc_calls,
                    "heuristic_calls": stats.heuristic_calls,
                    "fifty_fifty_clicks": stats.fifty_fifty_clicks,
                })
                completed += 1
                if progress_every and completed % progress_every == 0:
                    print(f"[{completed}/{total}] {completed/total:.1%} complete", flush=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Completed {total} games in {time.perf_counter()-overall_start:.2f}s.")
    print(f"Saved results to {output_csv}")
    return rows


def summarize_csv(input_csv: str) -> list[dict]:
    with open(input_csv, newline="", encoding="utf-8") as f:
        data = list(csv.DictReader(f))
    if not data:
        return []

    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in data:
        grouped.setdefault((row["configuration"], row["solver"]), []).append(row)

    summary = []
    for (configuration, solver), group in sorted(grouped.items()):
        n = len(group)
        wins = sum(int(x["won"]) for x in group)
        summary.append({
            "configuration": configuration,
            "solver": solver,
            "games": n,
            "wins": wins,
            "win_rate": wins / n,
            "avg_moves": mean(float(x["moves"]) for x in group),
            "avg_time_s": mean(float(x["elapsed_seconds"]) for x in group),
            "avg_max_frontier": mean(float(x["max_frontier"]) for x in group),
            "avg_exact_calls": mean(float(x["exact_calls"]) for x in group),
            "avg_mcmc_calls": mean(float(x["mcmc_calls"]) for x in group),
            "avg_heuristic_calls": mean(float(x["heuristic_calls"]) for x in group),
            "avg_50_50_clicks": mean(float(x["fifty_fifty_clicks"]) for x in group),
        })
    return summary
