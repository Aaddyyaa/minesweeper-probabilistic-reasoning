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
    heuristic_calls: int
    fifty_fifty_clicks: int
    method_history: list[str]


class RuleBasedAgent:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.solver = DeterministicSolver()

    def play(self, board: MinesweeperBoard, first_click: Cell | None = None) -> GameStats:
        start = time.perf_counter()
        if first_click is None:
            first_click = (board.rows // 2, board.cols // 2)
        board.reveal(first_click)
        moves = 1
        history: list[str] = []
        while not board.won and not board.exploded:
            safe, mines = self.solver.infer(board)
            for cell in sorted(mines):
                if cell not in board.flagged and cell not in board.revealed:
                    board.toggle_flag(cell)
            progressed = False
            for cell in sorted(safe - board.flagged):
                if cell in board.revealed:
                    continue
                alive, _ = board.reveal(cell)
                moves += 1
                progressed = True
                if not alive:
                    break
            if board.won or board.exploded:
                break
            guess = self.solver.choose_guess(board, self.rng)
            if guess is None:
                break
            history.append("random")
            alive, _ = board.reveal(guess)
            moves += 1
            if not alive:
                break
            if not progressed:
                history.append("forced-guess")
        return GameStats(
            board.won, moves, time.perf_counter() - start, 0, 0, 0, 0, 0, history
        )


class ProbabilisticAgent:
    def __init__(
        self,
        seed: int | None = None,
        exact_component_cap: int = 28,
        exact_max_seconds: float = 1.5,
        exact_max_nodes: int = 1_000_000,
        mcmc_samples: int = 3000,
        utility: bool = False,
    ) -> None:
        self.rng = random.Random(seed)
        self.exact = ExactProbabilitySolver(
            max_component_variables=exact_component_cap,
            max_seconds=exact_max_seconds,
            max_nodes=exact_max_nodes,
        )
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

    def play(self, board: MinesweeperBoard, first_click: Cell | None = None) -> GameStats:
        start = time.perf_counter()
        if first_click is None:
            first_click = (board.rows // 2, board.cols // 2)
        board.reveal(first_click)
        moves = 1
        max_frontier = 0
        exact_calls = mcmc_calls = heuristic_calls = fifty = 0
        history: list[str] = []

        while not board.won and not board.exploded:
            inference = self.infer(board)
            history.append(inference.method)
            max_frontier = max(max_frontier, len(inference.frontier_cells))
            if inference.method == "exact":
                exact_calls += 1
            elif inference.method.startswith("mcmc"):
                mcmc_calls += 1
            else:
                heuristic_calls += 1

            for cell, p in inference.probabilities.items():
                if p >= 1.0 - 1e-12 and cell not in board.flagged and cell not in board.revealed:
                    board.toggle_flag(cell)

            chosen = self.policy.choose(board, inference)
            if chosen is None:
                break
            p_chosen = inference.probabilities.get(chosen, 1.0)
            if abs(p_chosen - 0.5) < 1e-6:
                fifty += 1
            alive, _ = board.reveal(chosen)
            moves += 1
            if not alive:
                break

        return GameStats(
            board.won, moves, time.perf_counter() - start, max_frontier,
            exact_calls, mcmc_calls, heuristic_calls, fifty, history
        )
