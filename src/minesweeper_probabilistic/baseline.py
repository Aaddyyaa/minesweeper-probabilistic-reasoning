from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .board import Cell, MinesweeperBoard
from .constraints import Constraint, extract_constraints


@dataclass
class DeterministicSolver:
    """Rule-based baseline: clue saturation + subset-difference deductions, then random guessing."""
    name: str = "deterministic"

    def infer(self, board: MinesweeperBoard) -> tuple[set[Cell], set[Cell]]:
        constraints, _, _ = extract_constraints(board)
        safe: set[Cell] = set()
        mines: set[Cell] = set()

        for c in constraints:
            if c.mines == 0:
                safe.update(c.cells)
            elif c.mines == len(c.cells):
                mines.update(c.cells)

        # 1-1 / 1-2 style reasoning expressed as subset differences.
        changed = True
        current = list(constraints)
        while changed:
            changed = False
            additions: list[Constraint] = []
            for a in current:
                for b in current:
                    if a is b or not a.cells.issubset(b.cells):
                        continue
                    diff_cells = b.cells - a.cells
                    diff_mines = b.mines - a.mines
                    if not diff_cells or not 0 <= diff_mines <= len(diff_cells):
                        continue
                    new_c = Constraint(frozenset(diff_cells), diff_mines)
                    if new_c not in current and new_c not in additions:
                        additions.append(new_c)
                    if diff_mines == 0:
                        before = len(safe)
                        safe.update(diff_cells)
                        changed |= len(safe) != before
                    elif diff_mines == len(diff_cells):
                        before = len(mines)
                        mines.update(diff_cells)
                        changed |= len(mines) != before
            if additions:
                current.extend(additions)
                changed = True

        # Never return a contradictory recommendation.
        mines.difference_update(safe)
        return safe, mines

    def choose_guess(self, board: MinesweeperBoard, rng) -> Optional[Cell]:
        hidden = sorted(board.unflagged_hidden_cells() - set(board.revealed))
        return rng.choice(hidden) if hidden else None
