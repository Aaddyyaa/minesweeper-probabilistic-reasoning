from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Optional

from .board import Cell, MinesweeperBoard
from .exact_solver import InferenceResult


def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * log2(p) + (1.0 - p) * log2(1.0 - p))


@dataclass
class SafetyPolicy:
    name: str = "lowest-risk"

    def choose(self, board: MinesweeperBoard, inference: InferenceResult) -> Optional[Cell]:
        candidates = sorted(board.unflagged_hidden_cells())
        if not candidates:
            return None
        certain_safe = [
            c for c in candidates if inference.probabilities.get(c, 1.0) <= 1e-12
        ]
        if certain_safe:
            return certain_safe[0]
        return min(
            candidates,
            key=lambda c: (inference.probabilities.get(c, 1.0), c),
        )


@dataclass
class UtilityPolicy:
    """Risk + information heuristic.

    The terminal utility of a click is 1-2p for win=+1 and loss=-1.
    The information term is a transparent entropy proxy, not exact value-of-information.
    """

    information_weight: float = 0.04
    name: str = "risk-information-utility"

    def choose(self, board: MinesweeperBoard, inference: InferenceResult) -> Optional[Cell]:
        candidates = sorted(board.unflagged_hidden_cells())
        if not candidates:
            return None

        safe = [c for c in candidates if inference.probabilities.get(c, 1.0) <= 1e-12]
        if safe:
            return max(
                safe,
                key=lambda c: (self._information_score(board, inference, c), -c[0], -c[1]),
            )

        best_cell = None
        best_score = float("-inf")
        for cell in candidates:
            p = inference.probabilities.get(cell, 1.0)
            expected_utility = 1.0 - 2.0 * p
            score = expected_utility + self.information_weight * self._information_score(
                board, inference, cell
            )
            if score > best_score:
                best_score = score
                best_cell = cell
        return best_cell

    @staticmethod
    def _information_score(
        board: MinesweeperBoard, inference: InferenceResult, cell: Cell
    ) -> float:
        score = _binary_entropy(inference.probabilities.get(cell, 0.5))
        for neighbor in board.neighbors(cell):
            if neighbor not in board.revealed and neighbor not in board.flagged:
                score += 0.5 * _binary_entropy(
                    inference.probabilities.get(neighbor, 0.5)
                )
        return score
