from __future__ import annotations

import argparse
import csv
import os
import random

from .agent import ProbabilisticAgent, RuleBasedAgent
from .board import MinesweeperBoard
from .experiment import run_experiments, summarize_csv
from .visualize import save_factor_graph, save_probability_heatmap, save_runtime_chart, save_win_rate_chart


def play_interactive(rows: int, cols: int, mines: int) -> None:
    board = MinesweeperBoard(rows, cols, mines)
    print("Commands: r ROW COL = reveal, f ROW COL = flag, q = quit")
    while not board.won and not board.exploded:
        print(board.render())
        command = input("> ").strip().split()
        if not command:
            continue
        if command[0].lower() == "q":
            return
        if len(command) != 3 or command[0].lower() not in {"r", "f"}:
            print("Use r row col or f row col")
            continue
        r, c = int(command[1]), int(command[2])
        if command[0].lower() == "r":
            board.reveal((r, c))
        else:
            board.toggle_flag((r, c))
    print(board.render(show_mines=True))
    print("YOU WIN" if board.won else "BOOM")


def make_demo_outputs() -> None:
    board = MinesweeperBoard(9, 9, 10, seed=20261008)
    agent = ProbabilisticAgent(seed=99, exact_component_cap=64)
    board.reveal((4, 4))
    for _ in range(4):
        inference = agent.infer(board)
        choice = agent.policy.choose(board, inference)
        if choice is None:
            break
        board.reveal(choice)
    inference = agent.infer(board)
    save_probability_heatmap(board, inference.probabilities, "results/probability_heatmap.png")
    save_factor_graph(board, "results/factor_graph.png")
    print(board.render())
    print("Frontier cells:", len(inference.frontier_cells))
    print("Independent components:", [len(c) for c in inference.components])
    print("Exact configurations:", inference.total_configurations)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minesweeper as probabilistic reasoning")
    sub = p.add_subparsers(dest="cmd", required=True)

    game = sub.add_parser("game", help="Play Minesweeper manually")
    game.add_argument("--rows", type=int, default=9)
    game.add_argument("--cols", type=int, default=9)
    game.add_argument("--mines", type=int, default=10)

    exp = sub.add_parser("experiment", help="Run solver comparison")
    exp.add_argument("--games", type=int, default=100)
    exp.add_argument("--output", default="results/experiment_results.csv")

    summary = sub.add_parser("summary", help="Summarize a results CSV")
    summary.add_argument("--input", default="results/experiment_results.csv")

    demo = sub.add_parser("demo", help="Generate heatmap and factor graph demo")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.cmd == "game":
        play_interactive(args.rows, args.cols, args.mines)
    elif args.cmd == "experiment":
        run_experiments(args.games, args.output)
        print(f"Saved {args.output}")
        save_win_rate_chart(args.output, "results/win_rate_comparison.png")
        save_runtime_chart(args.output, "results/runtime_comparison.png")
    elif args.cmd == "summary":
        for row in summarize_csv(args.input):
            print(row)
    elif args.cmd == "demo":
        make_demo_outputs()
