from __future__ import annotations

import random
from dataclasses import dataclass

from .board import Cell, MinesweeperBoard
from .constraints import Constraint, extract_constraints
from .exact_solver import InferenceResult


@dataclass
class MCMCSolver:
    samples: int = 8000
    burn_in: int = 2000
    thinning: int = 5
    restarts: int = 8
    max_initial_nodes: int = 250_000
    seed: int | None = None

    def infer(self, board: MinesweeperBoard) -> InferenceResult:
        constraints, frontier, interior = extract_constraints(board)
        hidden = sorted(board.unflagged_hidden_cells())
        remaining_mines = board.mines - len(board.flagged)
        probabilities: dict[Cell, float] = {cell: 0.0 for cell in board.revealed}
        probabilities.update({cell: 1.0 for cell in board.flagged})

        if not hidden:
            return InferenceResult(probabilities, False, frontier, interior, [], 1, "mcmc")

        rng = random.Random(self.seed)
        assignment = self._find_valid_assignment(hidden, remaining_mines, constraints, rng)
        if assignment is None:
            raise ValueError("MCMC could not find an initial state satisfying the revealed clues")

        mine_set = set(cell for cell, value in assignment.items() if value)
        all_cells = list(hidden)
        mine_cells = [c for c in all_cells if c in mine_set]
        safe_cells = [c for c in all_cells if c not in mine_set]
        counts = {cell: 0 for cell in all_cells}

        accepted = 0
        total_iterations = self.burn_in + self.samples * self.thinning
        for step in range(total_iterations):
            if mine_cells and safe_cells:
                mine_cell = rng.choice(mine_cells)
                safe_cell = rng.choice(safe_cells)
                candidate_mines = set(mine_set)
                candidate_mines.remove(mine_cell)
                candidate_mines.add(safe_cell)
                if self._satisfies(candidate_mines, constraints):
                    mine_set = candidate_mines
                    mine_cells.remove(mine_cell)
                    mine_cells.append(safe_cell)
                    safe_cells.remove(safe_cell)
                    safe_cells.append(mine_cell)
                    accepted += 1

            if step >= self.burn_in and (step - self.burn_in) % self.thinning == 0:
                for cell in all_cells:
                    counts[cell] += cell in mine_set

        used_samples = max(1, self.samples)
        for cell in all_cells:
            # Laplace smoothing prevents a finite chain from producing false certainty.
            probabilities[cell] = (counts[cell] + 0.5) / (used_samples + 1.0)

        return InferenceResult(
            probabilities=probabilities,
            exact=False,
            frontier_cells=frontier,
            interior_cells=interior,
            components=[],
            total_configurations=accepted,
            method=f"mcmc ({accepted} accepted moves)",
        )

    def _find_valid_assignment(
        self,
        cells: list[Cell],
        mine_count: int,
        constraints: list[Constraint],
        rng: random.Random,
    ) -> dict[Cell, int] | None:
        """Find one globally valid state without enumerating every solution.

        The search is over frontier variables only; interior mines are filled
        arbitrarily after a feasible frontier assignment is found.
        """
        frontier = sorted({cell for c in constraints for cell in c.cells})
        interior = [cell for cell in cells if cell not in frontier]
        min_frontier_mines = max(0, mine_count - len(interior))
        max_frontier_mines = min(mine_count, len(frontier))
        if not min_frontier_mines <= max_frontier_mines:
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
            possible_max = mines_so_far + (len(order) - pos)
            if possible_max < min_frontier_mines:
                return False
            if pos == len(order):
                if not (min_frontier_mines <= mines_so_far <= max_frontier_mines):
                    return False
                return all(feasible(ci) for ci in range(len(local)))

            vi = order[pos]
            values = [0, 1]
            rng.shuffle(values)
            for value in values:
                assignment[vi] = value
                for ci in cell_constraints[vi]:
                    assigned_count[ci] += 1
                    assigned_mines[ci] += value
                if all(feasible(ci) for ci in cell_constraints[vi]) and backtrack(pos + 1, mines_so_far + value):
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
        return all(len(mine_set & set(c.cells)) == c.mines for c in constraints)


class ConstraintHeuristicSolver:
    """Fast fractional heuristic used only if exact/MCMC initialization is exhausted."""

    def infer(self, board: MinesweeperBoard) -> InferenceResult:
        constraints, frontier, interior = extract_constraints(board)
        hidden = board.unflagged_hidden_cells()
        remaining_mines = max(0, board.mines - len(board.flagged))
        probabilities = {cell: 0.0 for cell in board.revealed}
        probabilities.update({cell: 1.0 for cell in board.flagged})
        if not hidden:
            return InferenceResult(probabilities, False, frontier, interior, [], 0, "constraint-heuristic")

        global_prior = remaining_mines / len(hidden)
        p = {cell: global_prior for cell in hidden}
        for _ in range(12):
            updates = {cell: [] for cell in hidden}
            for c in constraints:
                denom = sum(max(p[cell], 1e-9) for cell in c.cells)
                for cell in c.cells:
                    share = p[cell] / denom if denom else 1.0 / len(c.cells)
                    # Allocate the clue's remaining mines proportionally to current belief.
                    updates[cell].append(c.mines * share)
            for cell in frontier:
                local = sum(updates[cell]) / len(updates[cell]) if updates[cell] else global_prior
                p[cell] = min(1.0, max(0.0, 0.65 * local + 0.35 * global_prior))
            # Keep the expected total mine count aligned with the global budget.
            scale = remaining_mines / max(sum(p.values()), 1e-12)
            for cell in hidden:
                p[cell] = min(1.0, max(0.0, p[cell] * scale))

        probabilities.update(p)
        return InferenceResult(probabilities, False, frontier, interior, [], 0, "constraint-heuristic")
