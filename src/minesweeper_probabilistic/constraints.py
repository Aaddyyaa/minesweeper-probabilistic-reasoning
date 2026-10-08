from __future__ import annotations

from dataclasses import dataclass

from .board import Cell, MinesweeperBoard


@dataclass(frozen=True)
class Constraint:
    cells: frozenset[Cell]
    mines: int

    def __post_init__(self) -> None:
        if self.mines < 0 or self.mines > len(self.cells):
            raise ValueError("Invalid constraint")


def extract_constraints(board: MinesweeperBoard) -> tuple[list[Constraint], set[Cell], set[Cell]]:
    """Return constraints, frontier cells, and interior cells among hidden cells."""
    if board.first_click is None:
        hidden = board.unflagged_hidden_cells()
        return [], set(), hidden

    constraints: list[Constraint] = []
    frontier: set[Cell] = set()
    for clue in board.revealed:
        unknown = board.unflagged_hidden_cells().intersection(board.neighbors(clue))
        if not unknown:
            continue
        flagged_near = len(set(board.neighbors(clue)).intersection(board.flagged))
        remaining = board.number(clue) - flagged_near
        if 0 <= remaining <= len(unknown):
            constraints.append(Constraint(frozenset(unknown), remaining))
            frontier.update(unknown)
        else:
            raise ValueError(
                f"Inconsistent board state around {clue}: clue={board.number(clue)}, "
                f"flagged={flagged_near}, unknown={len(unknown)}"
            )

    hidden = board.unflagged_hidden_cells()
    interior = hidden - frontier
    return constraints, frontier, interior


def connected_components(frontier: set[Cell], constraints: list[Constraint]) -> list[set[Cell]]:
    """Split frontier variables into independent components by shared constraints."""
    if not frontier:
        return []

    graph: dict[Cell, set[Cell]] = {cell: set() for cell in frontier}
    for constraint in constraints:
        cells = list(constraint.cells)
        for i, cell in enumerate(cells):
            graph[cell].update(c for c in cells[i + 1 :])
            for other in cells[:i]:
                graph[cell].add(other)

    components: list[set[Cell]] = []
    unseen = set(frontier)
    while unseen:
        start = next(iter(unseen))
        stack = [start]
        comp: set[Cell] = set()
        while stack:
            cell = stack.pop()
            if cell not in unseen:
                continue
            unseen.remove(cell)
            comp.add(cell)
            stack.extend(graph[cell] & unseen)
        components.append(comp)
    return components
