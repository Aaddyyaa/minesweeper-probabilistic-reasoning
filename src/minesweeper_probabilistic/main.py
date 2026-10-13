from __future__ import annotations

import argparse
import json

from .agent import ProbabilisticAgent
from .board import MinesweeperBoard
from .experiment import run_experiments, summarize_csv
from .visualize import (
    save_factor_graph,
    save_probability_heatmap,
    save_runtime_chart,
    save_win_rate_chart,
)


def play_interactive(rows: int, cols: int, mines: int) -> None:
    board = MinesweeperBoard(rows, cols, mines)
    print("Commands: r ROW COL = reveal, f ROW COL = flag, q = quit")
    while not board.won and not board.exploded:
        print(board.render())
        command = input("> ").strip().split()
        if not command:
            continue
        try:
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
        except ValueError as exc:
            print(f"Invalid command: {exc}")
    print(board.render(show_mines=True))
    print("YOU WIN" if board.won else "BOOM")


def make_demo_outputs() -> None:
    board = MinesweeperBoard(9, 9, 10, seed=20261008)
    agent = ProbabilisticAgent(seed=99)
    board.reveal((4, 4))
    for _ in range(5):
        inference = agent.infer(board)
        choice = agent.policy.choose(board, inference)
        if choice is None:
            break
        board.reveal(choice)

    inference = agent.infer(board)
    save_probability_heatmap(
        board, inference.probabilities, "results/probability_heatmap.png"
    )
    save_factor_graph(board, "results/factor_graph.png")

    print(board.render(probabilities=inference.probabilities))
    print("Frontier cells:", len(inference.frontier_cells))
    print("Independent components:", [len(c) for c in inference.components])
    print("Inference method:", inference.method)
    print("Exact configurations:", inference.total_configurations)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minesweeper as probabilistic reasoning"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    game = sub.add_parser("game", help="Play Minesweeper manually")
    game.add_argument("--rows", type=int, default=9)
    game.add_argument("--cols", type=int, default=9)
    game.add_argument("--mines", type=int, default=10)

    exp = sub.add_parser("experiment", help="Run solver comparison")
    exp.add_argument("--games", type=int, default=100)
    exp.add_argument("--output", default="results/experiment_results.csv")
    exp.add_argument("--seed-start", type=int, default=20261008)
    exp.add_argument("--progress-every", type=int, default=25)

    summary = sub.add_parser("summary", help="Summarize a results CSV")
    summary.add_argument("--input", default="results/experiment_results.csv")

    demo = sub.add_parser("demo", help="Generate heatmap and factor graph")

    inspect = sub.add_parser(
        "inspect", help="Print posterior probabilities for a reproducible board state"
    )
    inspect.add_argument("--rows", type=int, default=9)
    inspect.add_argument("--cols", type=int, default=9)
    inspect.add_argument("--mines", type=int, default=10)
    inspect.add_argument("--seed", type=int, default=20261008)
    inspect.add_argument("--row", type=int, default=-1)
    inspect.add_argument("--col", type=int, default=-1)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.cmd == "game":
        play_interactive(args.rows, args.cols, args.mines)
    elif args.cmd == "experiment":
        run_experiments(
            games_per_config=args.games,
            output_csv=args.output,
            seed_start=args.seed_start,
            progress_every=args.progress_every,
        )
        save_win_rate_chart(args.output, "results/win_rate_comparison.png")
        save_runtime_chart(args.output, "results/runtime_comparison.png")
    elif args.cmd == "summary":
        summary = summarize_csv(args.input)
        for row in summary:
            print(json.dumps(row, sort_keys=True))
    elif args.cmd == "demo":
        make_demo_outputs()
    elif args.cmd == "inspect":
        board = MinesweeperBoard(args.rows, args.cols, args.mines, seed=args.seed)
        cell = (
            (args.row, args.col)
            if args.row >= 0 and args.col >= 0
            else (args.rows // 2, args.cols // 2)
        )
        board.reveal(cell)
        result = ProbabilisticAgent(seed=args.seed).infer(board)
        print(board.render(probabilities=result.probabilities))
        print(json.dumps(
            {
                "method": result.method,
                "frontier": len(result.frontier_cells),
                "interior": len(result.interior_cells),
                "components": [len(c) for c in result.components],
                "total_configurations": result.total_configurations,
            },
            indent=2,
        ))


if __name__ == "__main__":
    main()
