from __future__ import annotations

from dataclasses import dataclass
import random
import time

from .baseline import DeterministicSolver
from .board import Cell, MinesweeperBoard
from .exact_solver import ExactProbabilitySolver, InferenceResult
from .mcmc_solver import ConstraintHeuristicSolver, MCMCSolver
from .policy import SafetyPolicy, UtilityPolicy


@dataclass
class GameStats:
    won: bool
    moves: int
    elapsed_seconds: float
    max_frontier: int
    exact_calls: int
    mcmc_calls: int
    fifty_fifty_clicks: int
    method_history: list[str]


class RuleBasedAgent:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.solver = DeterministicSolver()

    def play(self, board: MinesweeperBoard, first_click: Cell = (0, 0)) -> GameStats:
        start = time.perf_counter()
        board.reveal(first_click)
        moves = 1
        history: list[str] = []
        max_frontier = 0
        fifty = 0
        while not board.won and not board.exploded:
            safe, mines = self.solver.infer(board)
            for cell in mines:
                if cell not in board.flagged:
                    board.toggle_flag(cell)
            for cell in sorted(safe - board.flagged):
                if cell in board.revealed:
                    continue
                alive, _ = board.reveal(cell)
                moves += 1
                if not alive:
                    break
            if board.won or board.exploded:
                break
            guess = self.solver.choose_guess(board, self.rng)
            if guess is None:
                break
            alive, _ = board.reveal(guess)
            moves += 1
            history.append("random")
            if not alive:
                break
        return GameStats(board.won, moves, time.perf_counter() - start, max_frontier, 0, 0, fifty, history)


class ProbabilisticAgent:
    def __init__(
        self,
        seed: int | None = None,
        exact_component_cap: int = 64,
        mcmc_samples: int = 5000,
        utility: bool = False,
    ) -> None:
        self.rng = random.Random(seed)
        self.exact = ExactProbabilitySolver(exact_component_cap)
        self.mcmc = MCMCSolver(samples=mcmc_samples, seed=seed)
        self.policy = UtilityPolicy() if utility else SafetyPolicy()
        self.heuristic = ConstraintHeuristicSolver()
        self.utility = utility

    def infer(self, board: MinesweeperBoard) -> InferenceResult:
        try:
            return self.exact.infer(board)
        except RuntimeError:
            try:
                return self.mcmc.infer(board)
            except (RuntimeError, ValueError):
                return self.heuristic.infer(board)

    def play(self, board: MinesweeperBoard, first_click: Cell = (0, 0)) -> GameStats:
        start = time.perf_counter()
        board.reveal(first_click)
        moves = 1
        max_frontier = 0
        exact_calls = 0
        mcmc_calls = 0
        fifty = 0
        history: list[str] = []

        while not board.won and not board.exploded:
            inference = self.infer(board)
            history.append(inference.method)
            if inference.exact:
                exact_calls += 1
            else:
                mcmc_calls += 1
            max_frontier = max(max_frontier, len(inference.frontier_cells))

            for cell, p in inference.probabilities.items():
                if p >= 1.0 - 1e-12 and cell not in board.flagged and cell not in board.revealed:
                    board.toggle_flag(cell)

            candidates = board.unflagged_hidden_cells()
            if any(abs(inference.probabilities.get(c, 0.0) - 0.5) < 1e-9 for c in candidates):
                chosen = self.policy.choose(board, inference)
                if chosen is not None and abs(inference.probabilities.get(chosen, 0.0) - 0.5) < 1e-9:
                    fifty += 1

            chosen = self.policy.choose(board, inference)
            if chosen is None:
                break
            alive, _ = board.reveal(chosen)
            moves += 1
            if not alive:
                break

        return GameStats(board.won, moves, time.perf_counter() - start, max_frontier, exact_calls, mcmc_calls, fifty, history)
