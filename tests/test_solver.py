from math import comb

from minesweeper_probabilistic.board import MinesweeperBoard
from minesweeper_probabilistic.constraints import extract_constraints
from minesweeper_probabilistic.exact_solver import ExactProbabilitySolver
from minesweeper_probabilistic.mcmc_solver import MCMCSolver
from minesweeper_probabilistic.policy import UtilityPolicy


def test_first_click_safe() -> None:
    board = MinesweeperBoard(9, 9, 10, seed=1)
    alive, revealed = board.reveal((4, 4))
    assert alive
    assert (4, 4) not in board._mine_cells
    assert revealed


def test_constraint_extraction() -> None:
    board = MinesweeperBoard(3, 3, 1, seed=2)
    board.reveal((1, 1))
    constraints, frontier, interior = extract_constraints(board)
    assert constraints
    assert frontier
    assert interior == set()


def test_interior_uniform_probability_without_clues() -> None:
    board = MinesweeperBoard(4, 4, 4, seed=3)
    board.reveal((0, 0))
    solver = ExactProbabilitySolver()
    result = solver.infer(board)
    hidden = board.unflagged_hidden_cells()
    total_probability = sum(result.probabilities[c] for c in hidden)
    assert abs(total_probability - (board.mines - len(board.flagged))) < 1e-9


def test_probability_normalization_small_board() -> None:
    board = MinesweeperBoard(5, 5, 5, seed=4)
    board.reveal((2, 2))
    solver = ExactProbabilitySolver(max_component_variables=24)
    result = solver.infer(board)
    hidden = board.unflagged_hidden_cells()
    assert abs(sum(result.probabilities[c] for c in hidden) - 5) < 1e-9


def test_exact_global_weighting_changes_interior() -> None:
    board = MinesweeperBoard(6, 6, 6, seed=11)
    board.reveal((3, 3))
    result = ExactProbabilitySolver(max_component_variables=24).infer(board)
    hidden = board.unflagged_hidden_cells()
    assert abs(sum(result.probabilities[c] for c in hidden) - 6) < 1e-9
    if result.interior_cells:
        values = {round(result.probabilities[c], 12) for c in result.interior_cells}
        assert len(values) == 1


def test_zero_constraint_implies_safe() -> None:
    board = MinesweeperBoard(5, 5, 3, seed=5)
    board.reveal((0, 0))
    result = ExactProbabilitySolver(max_component_variables=24).infer(board)
    for clue in board.revealed:
        if board.number(clue) != 0:
            continue
        for n in board.neighbors(clue):
            if n in board.unflagged_hidden_cells():
                assert result.probabilities[n] == 0.0


def test_mcmc_exact_mine_budget_normalization() -> None:
    board = MinesweeperBoard(8, 8, 10, seed=7)
    board.reveal((4, 4))
    result = MCMCSolver(samples=300, burn_in=100, thinning=2, seed=7).infer(board)
    hidden = board.unflagged_hidden_cells()
    total = sum(result.probabilities[c] for c in hidden)
    assert abs(total - (board.mines - len(board.flagged))) < 1e-9
    assert all(0.0 <= result.probabilities[c] <= 1.0 for c in hidden)


def test_mcmc_preserves_global_mine_count() -> None:
    board = MinesweeperBoard(8, 8, 10, seed=7)
    board.reveal((4, 4))
    result = MCMCSolver(samples=500, burn_in=100, thinning=2, seed=7).infer(board)
    hidden = board.unflagged_hidden_cells()
    total = sum(result.probabilities[c] for c in hidden)
    assert abs(total - (board.mines - len(board.flagged))) < 0.2


def test_utility_policy_never_prefers_higher_risk_without_information() -> None:
    board = MinesweeperBoard(5, 5, 5, seed=8)
    board.reveal((2, 2))
    result = ExactProbabilitySolver(max_component_variables=24).infer(board)
    policy = UtilityPolicy(information_weight=0.0)
    chosen = policy.choose(board, result)
    minimum = min(result.probabilities[c] for c in board.unflagged_hidden_cells())
    assert result.probabilities[chosen] <= minimum + 1e-12
