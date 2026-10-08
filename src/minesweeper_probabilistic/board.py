from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Iterable, Iterator, Optional

Cell = tuple[int, int]


@dataclass
class MinesweeperBoard:
    rows: int
    cols: int
    mines: int
    seed: Optional[int] = None
    _mine_cells: set[Cell] = field(default_factory=set, init=False)
    _numbers: dict[Cell, int] = field(default_factory=dict, init=False)
    revealed: set[Cell] = field(default_factory=set, init=False)
    flagged: set[Cell] = field(default_factory=set, init=False)
    first_click: Optional[Cell] = field(default=None, init=False)
    exploded: bool = field(default=False, init=False)
    won: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("rows and cols must be positive")
        if not 0 <= self.mines < self.rows * self.cols:
            raise ValueError("mines must be in [0, rows*cols)")
        self._rng = random.Random(self.seed)

    @property
    def size(self) -> int:
        return self.rows * self.cols

    def cells(self) -> Iterator[Cell]:
        for r in range(self.rows):
            for c in range(self.cols):
                yield (r, c)

    def neighbors(self, cell: Cell) -> Iterator[Cell]:
        r, c = cell
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    yield (nr, nc)

    def _place_mines(self, safe_cell: Cell) -> None:
        candidates = [cell for cell in self.cells() if cell != safe_cell]
        self._mine_cells = set(self._rng.sample(candidates, self.mines))
        self._numbers.clear()
        for cell in self.cells():
            if cell in self._mine_cells:
                continue
            self._numbers[cell] = sum(n in self._mine_cells for n in self.neighbors(cell))

    def reveal(self, cell: Cell) -> tuple[bool, list[Cell]]:
        """Reveal a cell. Returns (alive, newly_revealed_cells). First click is safe."""
        if self.won or self.exploded:
            return not self.exploded, []
        if cell in self.flagged or cell in self.revealed:
            return True, []
        if not self.in_bounds(cell):
            raise ValueError(f"out-of-bounds cell: {cell}")

        if self.first_click is None:
            self.first_click = cell
            self._place_mines(cell)

        if cell in self._mine_cells:
            self.exploded = True
            self.revealed.add(cell)
            return False, [cell]

        newly_revealed: list[Cell] = []
        stack = [cell]
        seen: set[Cell] = set()
        while stack:
            current = stack.pop()
            if current in seen or current in self.revealed or current in self.flagged:
                continue
            seen.add(current)
            if current in self._mine_cells:
                continue
            self.revealed.add(current)
            newly_revealed.append(current)
            if self._numbers[current] == 0:
                stack.extend(self.neighbors(current))

        self._update_win_state()
        return True, newly_revealed

    def toggle_flag(self, cell: Cell) -> bool:
        if cell in self.revealed or self.exploded or self.won:
            return False
        if cell in self.flagged:
            self.flagged.remove(cell)
        else:
            self.flagged.add(cell)
        return True

    def number(self, cell: Cell) -> int:
        if cell not in self._numbers:
            raise ValueError("Number is unavailable before the first click generates the board")
        return self._numbers[cell]

    def is_mine(self, cell: Cell) -> bool:
        return cell in self._mine_cells

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def hidden_cells(self) -> set[Cell]:
        return set(self.cells()) - self.revealed

    def unflagged_hidden_cells(self) -> set[Cell]:
        return self.hidden_cells() - self.flagged

    def _update_win_state(self) -> None:
        safe_total = self.size - self.mines
        if len(self.revealed) >= safe_total and not self.exploded:
            self.won = True

    def render(self, show_mines: bool = False, probabilities: Optional[dict[Cell, float]] = None) -> str:
        lines: list[str] = []
        border = "+" + "---" * self.cols + "+"
        lines.append(border)
        for r in range(self.rows):
            row = ["|"]
            for c in range(self.cols):
                cell = (r, c)
                if show_mines and cell in self._mine_cells:
                    value = " * "
                elif cell in self.flagged:
                    value = " F "
                elif cell not in self.revealed:
                    if probabilities and cell in probabilities:
                        value = f"{probabilities[cell]:.1f}"[-3:]
                    else:
                        value = " # "
                else:
                    value = f" {self._numbers[cell]} "
                row.append(value + "|")
            lines.append("".join(row))
            lines.append(border)
        return "\n".join(lines)

    def state_copy(self) -> "MinesweeperBoard":
        other = MinesweeperBoard(self.rows, self.cols, self.mines, self.seed)
        other._rng.setstate(self._rng.getstate())
        other._mine_cells = set(self._mine_cells)
        other._numbers = dict(self._numbers)
        other.revealed = set(self.revealed)
        other.flagged = set(self.flagged)
        other.first_click = self.first_click
        other.exploded = self.exploded
        other.won = self.won
        return other
