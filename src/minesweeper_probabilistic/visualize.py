from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from .board import MinesweeperBoard
from .constraints import extract_constraints


SOLVER_LABELS = {
    "baseline": "Baseline",
    "probabilistic_safety": "Probabilistic safety",
    "probabilistic_utility": "Probabilistic utility",
}


def probability_matrix(board: MinesweeperBoard, probabilities: dict[tuple[int, int], float]) -> np.ndarray:
    matrix = np.full((board.rows, board.cols), np.nan, dtype=float)
    for (r, c), p in probabilities.items():
        matrix[r, c] = p
    return matrix


def save_probability_heatmap(
    board: MinesweeperBoard,
    probabilities: dict[tuple[int, int], float],
    path: str,
    title: str = "Posterior mine probability",
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    matrix = probability_matrix(board, probabilities)
    fig, ax = plt.subplots(figsize=(max(7, board.cols * 0.35), max(6, board.rows * 0.35)))
    im = ax.imshow(matrix, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(np.arange(-0.5, board.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, board.rows, 1), minor=True)
    ax.grid(which="minor", linewidth=0.35, alpha=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)
    for r in range(board.rows):
        for c in range(board.cols):
            cell = (r, c)
            if cell in board.revealed:
                label = str(board.number(cell))
            elif cell in board.flagged:
                label = "F"
            elif cell in probabilities:
                label = f"{probabilities[cell]:.2f}"
            else:
                continue
            ax.text(c, r, label, ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="P(cell is a mine)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _read_experiment_data(input_csv: str) -> list[dict[str, str]]:
    with open(input_csv, newline="", encoding="utf-8") as f:
        data = list(csv.DictReader(f))
    if not data:
        raise ValueError(f"No experiment rows found in {input_csv}")
    required = {"configuration", "solver", "won", "elapsed_seconds"}
    missing = required.difference(data[0])
    if missing:
        raise ValueError(f"Experiment CSV is missing columns: {', '.join(sorted(missing))}")
    return data


def _configuration_order(data: list[dict[str, str]]) -> list[str]:
    # Show only configurations that actually occur in the CSV; easy-only runs
    # therefore produce a single easy category rather than empty medium/hard bars.
    preferred = ["easy", "medium", "hard"]
    present = {row["configuration"] for row in data}
    return [name for name in preferred if name in present] + sorted(present.difference(preferred))


def save_win_rate_chart(input_csv: str, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = _read_experiment_data(input_csv)
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in data:
        grouped[(row["configuration"], row["solver"])].append(int(row["won"]))

    configs = _configuration_order(data)
    solvers = [s for s in SOLVER_LABELS if any(row["solver"] == s for row in data)]
    x = np.arange(len(configs))
    width = min(0.7 / max(len(solvers), 1), 0.24)
    fig, ax = plt.subplots(figsize=(max(7, len(configs) * 3.0), 5.5))
    for i, solver in enumerate(solvers):
        rates = [
            float(np.mean(grouped[(cfg, solver)])) if grouped[(cfg, solver)] else np.nan
            for cfg in configs
        ]
        bars = ax.bar(x + (i - (len(solvers) - 1) / 2) * width, rates, width,
                      label=SOLVER_LABELS.get(solver, solver.replace("_", " ").title()))
        for bar, rate in zip(bars, rates):
            if np.isfinite(rate):
                ax.annotate(f"{rate:.1%}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{cfg.title()}\n(n={sum(1 for row in data if row['configuration'] == cfg and row['solver'] == solvers[0])})" for cfg in configs])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Win rate")
    ax.set_title("Minesweeper solver win rate — easy boards only" if configs == ["easy"] else "Minesweeper solver win rate by configuration")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_runtime_chart(input_csv: str, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = _read_experiment_data(input_csv)
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in data:
        grouped[(row["configuration"], row["solver"])].append(float(row["elapsed_seconds"]))

    configs = _configuration_order(data)
    solvers = [s for s in SOLVER_LABELS if any(row["solver"] == s for row in data)]
    x = np.arange(len(configs))
    width = min(0.7 / max(len(solvers), 1), 0.24)
    fig, ax = plt.subplots(figsize=(max(7, len(configs) * 3.0), 5.5))
    for i, solver in enumerate(solvers):
        times = [
            float(np.mean(grouped[(cfg, solver)])) if grouped[(cfg, solver)] else np.nan
            for cfg in configs
        ]
        bars = ax.bar(x + (i - (len(solvers) - 1) / 2) * width, times, width,
                      label=SOLVER_LABELS.get(solver, solver.replace("_", " ").title()))
        for bar, value in zip(bars, times):
            if np.isfinite(value):
                ax.annotate(f"{value:.3f}s", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([cfg.title() for cfg in configs])
    ax.set_ylabel("Average game time (seconds)")
    ax.set_title("Minesweeper solver runtime — easy boards only" if configs == ["easy"] else "Minesweeper solver runtime by configuration")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_factor_graph(board: MinesweeperBoard, path: str) -> None:
    """Draw a constraint factor graph: variable nodes -> clue-factor nodes."""
    import networkx as nx

    constraints, frontier, _ = extract_constraints(board)
    graph = nx.Graph()
    for cell in sorted(frontier):
        graph.add_node(("v", cell), bipartite=0)
    for i, constraint in enumerate(constraints):
        factor = ("f", i)
        graph.add_node(factor, bipartite=1)
        for cell in constraint.cells:
            graph.add_edge(("v", cell), factor)

    fig, ax = plt.subplots(figsize=(12, 8))
    variable_nodes = [n for n in graph.nodes if n[0] == "v"]
    factor_nodes = [n for n in graph.nodes if n[0] == "f"]
    pos = nx.spring_layout(graph, seed=7)
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.35)
    nx.draw_networkx_nodes(graph, pos, nodelist=variable_nodes, node_size=250)
    nx.draw_networkx_nodes(graph, pos, nodelist=factor_nodes, node_size=350, node_shape="s")
    labels = {}
    for node in variable_nodes:
        labels[node] = f"{node[1][0]},{node[1][1]}"
    for node in factor_nodes:
        labels[node] = f"f{node[1]}"
    nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=8)
    ax.set_title("Minesweeper Markov-network factor graph")
    ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
