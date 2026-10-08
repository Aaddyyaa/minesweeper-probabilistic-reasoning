from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .board import Cell, MinesweeperBoard
from .exact_solver import InferenceResult


@dataclass
class SafetyPolicy:
    name: str = "lowest-risk"

    def choose(self, board: MinesweeperBoard, inference: InferenceResult) -> Optional[Cell]:
        candidates = sorted(board.unflagged_hidden_cells())
        if not candidates:
            return None
        certain_safe = [c for c in candidates if inference.probabilities.get(c, 1.0) <= 1e-12]
        if certain_safe:
            return certain_safe[0]
        return min(candidates, key=lambda c: inference.probabilities.get(c, 1.0))


@dataclass
class UtilityPolicy:
    information_weight: float = 0.12
    name: str = "risk-information-utility"

    def choose(self, board: MinesweeperBoard, inference: InferenceResult) -> Optional[Cell]:
        candidates = sorted(board.unflagged_hidden_cells())
        if not candidates:
            return None
        safe = [c for c in candidates if inference.probabilities.get(c, 1.0) <= 1e-12]
        if safe:
            # Among guaranteed-safe cells, prefer one with a large local footprint.
            return max(safe, key=lambda c: self._information_score(board, c))

        best_cell = None
        best_utility = float("-inf")
        for cell in candidates:
            p_mine = inference.probabilities.get(cell, 1.0)
            expected_utility = (1.0 - p_mine) - p_mine  # win=+1, loss=-1 => 1-2p
            score = expected_utility + self.information_weight * self._information_score(board, cell)
            if score > best_utility:
                best_utility = score
                best_cell = cell
        return best_cell

    @staticmethod
    def _information_score(board: MinesweeperBoard, cell: Cell) -> float:
        # Cheap proxy for expected information gain: a successful click near many
        # unrevealed cells is more likely to expose useful constraints.
        hidden_neighbors = sum(1 for n in board.neighbors(cell) if n not in board.revealed and n not in board.flagged)
        clue_neighbors = sum(1 for n in board.neighbors(cell) if n in board.revealed)
        return hidden_neighbors + 0.25 * clue_neighbors
