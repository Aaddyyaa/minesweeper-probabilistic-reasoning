from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from .board import MinesweeperBoard
from .constraints import extract_constraints


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


def save_win_rate_chart(input_csv: str, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(input_csv, newline="", encoding="utf-8") as f:
        data = list(csv.DictReader(f))
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in data:
        grouped[(row["configuration"], row["solver"])].append(int(row["won"]))

    configs = ["easy", "medium", "hard"]
    solvers = ["baseline", "probabilistic_safety", "probabilistic_utility"]
    x = np.arange(len(configs))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, solver in enumerate(solvers):
        rates = [np.mean(grouped[(cfg, solver)]) if grouped[(cfg, solver)] else 0 for cfg in configs]
        ax.bar(x + (i - 1) * width, rates, width, label=solver)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Win rate")
    ax.set_title("Minesweeper solver comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_runtime_chart(input_csv: str, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(input_csv, newline="", encoding="utf-8") as f:
        data = list(csv.DictReader(f))
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in data:
        grouped[(row["configuration"], row["solver"])].append(float(row["elapsed_seconds"]))

    configs = ["easy", "medium", "hard"]
    solvers = ["baseline", "probabilistic_safety", "probabilistic_utility"]
    x = np.arange(len(configs))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, solver in enumerate(solvers):
        times = [np.mean(grouped[(cfg, solver)]) if grouped[(cfg, solver)] else 0 for cfg in configs]
        ax.bar(x + (i - 1) * width, times, width, label=solver)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("Average game time (s)")
    ax.set_title("Computation cost")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
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
    pos = {}
    pos.update(nx.spring_layout(graph, seed=7))
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
