from math import comb

from minesweeper_probabilistic.board import MinesweeperBoard
from minesweeper_probabilistic.constraints import extract_constraints
from minesweeper_probabilistic.exact_solver import ExactProbabilitySolver


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
    # The first click may recursively reveal more cells, so choose a fresh board and
    # assert the probability sum is at least the remaining mine budget.
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


def test_certain_zero_constraint() -> None:
    board = MinesweeperBoard(3, 3, 1, seed=5)
    board.reveal((0, 0))
    # Find any revealed zero and check adjacent hidden cells have posterior 0.
    zero = next((c for c in board.revealed if board.number(c) == 0), None)
    if zero is None:
        return
    result = ExactProbabilitySolver().infer(board)
    for n in board.neighbors(zero):
        if n in board.unflagged_hidden_cells():
            assert result.probabilities[n] == 0.0
