# 5-minute live demo

## Part 1 — Show the model
Run:

```bash
PYTHONPATH=src python -m minesweeper_probabilistic.main demo
```

Open `results/factor_graph.png` and explain that circles are hidden-cell variables and squares are clue factors.

## Part 2 — Show posterior reasoning
Open `results/probability_heatmap.png`.
Explain that each hidden cell has a number between 0 and 1 representing posterior mine probability. A 0 means the clues rule out a mine; 1 means every valid layout makes it a mine.

## Part 3 — Show the experiment
Run a small check first:

```bash
PYTHONPATH=src python -m minesweeper_probabilistic.main experiment --games 5 --output results/demo_results.csv
PYTHONPATH=src python -m minesweeper_probabilistic.main summary --input results/demo_results.csv
```

Then explain that the final experiment should use 300 games per board configuration.

## Part 4 — Explain the global constraint
Use an example where many frontier configurations remain possible. If a component consumes more mines, fewer mines can remain in the interior. The binomial factor therefore changes all cell marginals, even for cells not touching a clue.

## Part 5 — Close with the key message
"The program is not merely finding a legal move. It counts or samples plausible worlds, converts them into posterior probabilities, and chooses an action under uncertainty."
