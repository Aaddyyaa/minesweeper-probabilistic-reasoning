from __future__ import annotations

import random
from dataclasses import dataclass

from .board import Cell, MinesweeperBoard
from .constraints import Constraint, extract_constraints
from .exact_solver import InferenceResult


@dataclass
class MCMCSolver:
    """Constrained Metropolis-style sampler over globally valid mine layouts.

    Proposals swap one mine with one safe cell, preserving the exact global
    mine count. Because the target distribution is uniform over valid layouts,
    every accepted valid state has equal weight.
    """

    samples: int = 8000
    burn_in: int = 2000
    thinning: int = 5
    restarts: int = 4
    max_initial_nodes: int = 250_000
    seed: int | None = None

    def infer(self, board: MinesweeperBoard) -> InferenceResult:
        constraints, frontier, interior = extract_constraints(board)
        hidden = sorted(board.unflagged_hidden_cells())
        remaining_mines = board.mines - len(board.flagged)
        probabilities: dict[Cell, float] = {cell: 0.0 for cell in board.revealed}
        probabilities.update({cell: 1.0 for cell in board.flagged})

        if remaining_mines < 0 or remaining_mines > len(hidden):
            raise ValueError("Invalid remaining mine count")
        if not hidden:
            return InferenceResult(probabilities, False, frontier, interior, [], 1, "mcmc", 1)

        rng = random.Random(self.seed)
        assignment = self._find_valid_assignment(hidden, remaining_mines, constraints, rng)
        if assignment is None:
            raise ValueError("MCMC could not find a valid initial state")

        mine_set = {cell for cell, value in assignment.items() if value}
        all_cells = list(hidden)
        mine_cells = [c for c in all_cells if c in mine_set]
        safe_cells = [c for c in all_cells if c not in mine_set]
        counts = {cell: 0 for cell in all_cells}

        collected = 0
        accepted = 0
        iterations = self.burn_in + self.samples * self.thinning
        for step in range(iterations):
            if mine_cells and safe_cells:
                mine_cell = rng.choice(mine_cells)
                safe_cell = rng.choice(safe_cells)
                candidate = mine_set.copy()
                candidate.remove(mine_cell)
                candidate.add(safe_cell)
                if self._satisfies(candidate, constraints):
                    mine_set = candidate
                    mine_cells[mine_cells.index(mine_cell)] = safe_cell
                    safe_cells[safe_cells.index(safe_cell)] = mine_cell
                    accepted += 1

            if step >= self.burn_in and (step - self.burn_in) % self.thinning == 0:
                for cell in all_cells:
                    counts[cell] += int(cell in mine_set)
                collected += 1

        if collected == 0:
            raise RuntimeError("MCMC collected no samples")

        for cell in all_cells:
            probabilities[cell] = counts[cell] / collected

        return InferenceResult(
            probabilities=probabilities,
            exact=False,
            frontier_cells=frontier,
            interior_cells=interior,
            components=[],
            total_configurations=accepted,
            method=f"mcmc ({accepted} accepted proposals)",
            total_samples=collected,
        )

    def _find_valid_assignment(
        self,
        cells: list[Cell],
        mine_count: int,
        constraints: list[Constraint],
        rng: random.Random,
    ) -> dict[Cell, int] | None:
        frontier = sorted({cell for c in constraints for cell in c.cells})
        interior = [cell for cell in cells if cell not in frontier]
        min_frontier_mines = max(0, mine_count - len(interior))
        max_frontier_mines = min(mine_count, len(frontier))
        if min_frontier_mines > max_frontier_mines:
            return None

        index = {cell: i for i, cell in enumerate(frontier)}
        local = [[index[cell] for cell in c.cells] for c in constraints]
        targets = [c.mines for c in constraints]
        cell_constraints: list[list[int]] = [[] for _ in frontier]
        for ci, vars_ in enumerate(local):
            for vi in vars_:
                cell_constraints[vi].append(ci)

        assignment = [-1] * len(frontier)
        assigned_mines = [0] * len(local)
        assigned_count = [0] * len(local)
        visited = 0
        order = sorted(range(len(frontier)), key=lambda i: (-len(cell_constraints[i]), rng.random()))

        def feasible(ci: int) -> bool:
            unassigned = len(local[ci]) - assigned_count[ci]
            return assigned_mines[ci] <= targets[ci] <= assigned_mines[ci] + unassigned

        def backtrack(pos: int, mines_so_far: int) -> bool:
            nonlocal visited
            visited += 1
            if visited > self.max_initial_nodes:
                return False
            if mines_so_far > max_frontier_mines:
                return False
            if mines_so_far + (len(order) - pos) < min_frontier_mines:
                return False
            if pos == len(order):
                return min_frontier_mines <= mines_so_far <= max_frontier_mines and all(
                    feasible(ci) for ci in range(len(local))
                )

            vi = order[pos]
            values = [0, 1]
            rng.shuffle(values)
            for value in values:
                assignment[vi] = value
                for ci in cell_constraints[vi]:
                    assigned_count[ci] += 1
                    assigned_mines[ci] += value
                if all(feasible(ci) for ci in cell_constraints[vi]) and backtrack(
                    pos + 1, mines_so_far + value
                ):
                    return True
                for ci in cell_constraints[vi]:
                    assigned_count[ci] -= 1
                    assigned_mines[ci] -= value
            assignment[vi] = -1
            return False

        if not backtrack(0, 0):
            return None

        mine_set = {frontier[i] for i, value in enumerate(assignment) if value == 1}
        remaining_interior = mine_count - len(mine_set)
        if not 0 <= remaining_interior <= len(interior):
            return None
        mine_set.update(rng.sample(interior, remaining_interior))
        return {cell: int(cell in mine_set) for cell in cells}

    @staticmethod
    def _satisfies(mine_set: set[Cell], constraints: list[Constraint]) -> bool:
        return all(len(mine_set.intersection(c.cells)) == c.mines for c in constraints)


class ConstraintHeuristicSolver:
    """Last-resort fast heuristic when neither exact nor MCMC can initialize."""

    def infer(self, board: MinesweeperBoard) -> InferenceResult:
        constraints, frontier, interior = extract_constraints(board)
        hidden = board.unflagged_hidden_cells()
        remaining_mines = max(0, board.mines - len(board.flagged))
        probabilities = {cell: 0.0 for cell in board.revealed}
        probabilities.update({cell: 1.0 for cell in board.flagged})
        if not hidden:
            return InferenceResult(probabilities, False, frontier, interior, [], 0, "constraint-heuristic", 0)

        global_prior = remaining_mines / len(hidden)
        p = {cell: global_prior for cell in hidden}
        for _ in range(12):
            updates = {cell: [] for cell in hidden}
            for constraint in constraints:
                denom = sum(max(p[cell], 1e-12) for cell in constraint.cells)
                for cell in constraint.cells:
                    share = p[cell] / denom if denom else 1.0 / len(constraint.cells)
                    updates[cell].append(constraint.mines * share)
            for cell in frontier:
                local = sum(updates[cell]) / len(updates[cell]) if updates[cell] else global_prior
                p[cell] = min(1.0, max(0.0, 0.65 * local + 0.35 * global_prior))
            total = sum(p.values())
            if total:
                scale = remaining_mines / total
                for cell in hidden:
                    p[cell] = min(1.0, max(0.0, p[cell] * scale))

        probabilities.update(p)
        return InferenceResult(probabilities, False, frontier, interior, [], 0, "constraint-heuristic", 0)
