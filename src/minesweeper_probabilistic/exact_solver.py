from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import comb
import time
from typing import Optional

from .board import Cell, MinesweeperBoard
from .constraints import Constraint, connected_components, extract_constraints


@dataclass
class ComponentEnumeration:
    cells: tuple[Cell, ...]
    solution_count_by_mines: dict[int, int]
    mine_count_by_cell_and_mines: dict[Cell, dict[int, int]]


@dataclass
class InferenceResult:
    probabilities: dict[Cell, float]
    exact: bool
    frontier_cells: set[Cell]
    interior_cells: set[Cell]
    components: list[set[Cell]]
    total_configurations: int = 0
    method: str = "exact"


class ExactProbabilitySolver:
    def __init__(self, max_component_variables: int = 64, max_seconds: float | None = 2.0, max_nodes: int = 2_000_000) -> None:
        self.max_component_variables = max_component_variables
        self.max_seconds = max_seconds
        self.max_nodes = max_nodes

    def infer(self, board: MinesweeperBoard) -> InferenceResult:
        constraints, frontier, interior = extract_constraints(board)
        hidden = board.unflagged_hidden_cells()
        remaining_mines = board.mines - len(board.flagged)

        probabilities: dict[Cell, float] = {cell: 0.0 for cell in board.revealed}
        probabilities.update({cell: 1.0 for cell in board.flagged})

        if remaining_mines < 0 or remaining_mines > len(hidden):
            raise ValueError("Board state has an impossible remaining mine count")

        if not frontier:
            p = remaining_mines / len(interior) if interior else 0.0
            for cell in interior:
                probabilities[cell] = p
            total = comb(len(interior), remaining_mines) if 0 <= remaining_mines <= len(interior) else 0
            return InferenceResult(probabilities, True, frontier, interior, [], total, "exact")

        components = connected_components(frontier, constraints)
        enumerated: list[ComponentEnumeration] = []
        for component in components:
            if len(component) > self.max_component_variables:
                raise RuntimeError(
                    f"Frontier component has {len(component)} variables; exact enumeration cap is "
                    f"{self.max_component_variables}. Use MCMC fallback."
                )
            enumerated.append(self._enumerate_component(component, constraints))

        total_configurations = 0
        for mine_count_tuple in self._combine_counts(enumerated):
            frontier_mines = sum(mine_count_tuple)
            interior_mines = remaining_mines - frontier_mines
            if 0 <= interior_mines <= len(interior):
                ways = comb(len(interior), interior_mines)
                for comp, k in zip(enumerated, mine_count_tuple):
                    ways *= comp.solution_count_by_mines.get(k, 0)
                total_configurations += ways

        if total_configurations == 0:
            raise ValueError("No globally valid mine configuration exists for the current state")

        # Prefix/suffix distributions let us compute each component's global weight efficiently.
        prefix: list[dict[int, int]] = [{0: 1}]
        for comp in enumerated:
            prefix.append(self._convolve(prefix[-1], comp.solution_count_by_mines))
        suffix: list[dict[int, int]] = [{} for _ in range(len(enumerated) + 1)]
        suffix[-1] = {0: 1}
        for i in range(len(enumerated) - 1, -1, -1):
            suffix[i] = self._convolve(enumerated[i].solution_count_by_mines, suffix[i + 1])

        for i, comp in enumerate(enumerated):
            other = self._convolve(prefix[i], suffix[i + 1])
            for cell in comp.cells:
                numerator = 0
                per_k = comp.mine_count_by_cell_and_mines[cell]
                for k, ways_cell_mine in per_k.items():
                    for other_mines, other_ways in other.items():
                        leftover = remaining_mines - k - other_mines
                        if 0 <= leftover <= len(interior):
                            numerator += ways_cell_mine * other_ways * comb(len(interior), leftover)
                probabilities[cell] = numerator / total_configurations

        if interior:
            expected_leftover = 0.0
            for frontier_mines, ways_frontier in prefix[-1].items():
                leftover = remaining_mines - frontier_mines
                if 0 <= leftover <= len(interior):
                    expected_leftover += ways_frontier * comb(len(interior), leftover) * leftover
            probabilities_per_interior = expected_leftover / total_configurations / len(interior)
            for cell in interior:
                probabilities[cell] = probabilities_per_interior

        return InferenceResult(
            probabilities=probabilities,
            exact=True,
            frontier_cells=frontier,
            interior_cells=interior,
            components=components,
            total_configurations=total_configurations,
            method="exact",
        )

    def _enumerate_component(self, component: set[Cell], constraints: list[Constraint]) -> ComponentEnumeration:
        cells = tuple(sorted(component))
        local_constraints = [c for c in constraints if c.cells & component]
        index = {cell: i for i, cell in enumerate(cells)}
        c_vars = [[index[cell] for cell in c.cells] for c in local_constraints]
        c_targets = [c.mines for c in local_constraints]
        cell_to_constraints: list[list[int]] = [[] for _ in cells]
        for ci, vars_ in enumerate(c_vars):
            for vi in vars_:
                cell_to_constraints[vi].append(ci)

        # High-constraint-degree cells usually cause earlier pruning.
        order = sorted(range(len(cells)), key=lambda i: (-len(cell_to_constraints[i]), cells[i]))
        assignment = [-1] * len(cells)
        assigned_mines = [0] * len(c_vars)
        assigned_count = [0] * len(c_vars)
        start_time = time.perf_counter()
        visited_nodes = 0

        solution_count_by_mines: dict[int, int] = defaultdict(int)
        mine_count_by_cell_and_mines: dict[Cell, dict[int, int]] = {
            cell: defaultdict(int) for cell in cells
        }

        def feasible(ci: int) -> bool:
            remaining_slots = len(c_vars[ci]) - assigned_count[ci]
            current = assigned_mines[ci]
            target = c_targets[ci]
            return current <= target <= current + remaining_slots

        def backtrack(pos: int) -> None:
            nonlocal visited_nodes
            visited_nodes += 1
            if self.max_nodes is not None and visited_nodes > self.max_nodes:
                raise RuntimeError("Exact enumeration node budget exceeded; use MCMC fallback.")
            if self.max_seconds is not None and time.perf_counter() - start_time > self.max_seconds:
                raise RuntimeError("Exact enumeration time budget exceeded; use MCMC fallback.")
            if pos == len(order):
                total_mines = sum(assignment)
                solution_count_by_mines[total_mines] += 1
                for vi, value in enumerate(assignment):
                    if value == 1:
                        mine_count_by_cell_and_mines[cells[vi]][total_mines] += 1
                return

            vi = order[pos]
            for value in (0, 1):
                assignment[vi] = value
                touched = cell_to_constraints[vi]
                for ci in touched:
                    assigned_count[ci] += 1
                    assigned_mines[ci] += value
                if all(feasible(ci) for ci in touched):
                    backtrack(pos + 1)
                for ci in touched:
                    assigned_count[ci] -= 1
                    assigned_mines[ci] -= value
            assignment[vi] = -1

        backtrack(0)
        return ComponentEnumeration(
            cells=cells,
            solution_count_by_mines=dict(solution_count_by_mines),
            mine_count_by_cell_and_mines={
                cell: dict(counts) for cell, counts in mine_count_by_cell_and_mines.items()
            },
        )

    @staticmethod
    def _convolve(a: dict[int, int], b: dict[int, int]) -> dict[int, int]:
        result: dict[int, int] = defaultdict(int)
        for ka, va in a.items():
            for kb, vb in b.items():
                result[ka + kb] += va * vb
        return dict(result)

    def _combine_counts(self, components: list[ComponentEnumeration]):
        if not components:
            yield tuple()
            return
        # This function is kept lazy and is mainly useful for small component sets.
        counts = [sorted(comp.solution_count_by_mines) for comp in components]
        current = [0] * len(components)

        def rec(i: int):
            if i == len(components):
                yield tuple(current)
                return
            for k in counts[i]:
                current[i] = k
                yield from rec(i + 1)

        yield from rec(0)
