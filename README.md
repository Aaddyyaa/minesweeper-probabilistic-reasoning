# Minesweeper as a Probabilistic Reasoning Problem

A research/academic implementation that treats Minesweeper as inference under uncertainty instead of a pure logic puzzle.

## What this project demonstrates

- **Markov network / factor graph:** hidden cells are binary variables and revealed numbers become sum-constraint factors.
- **Independence:** the hidden frontier is split into connected components; components not linked by a common constraint can be solved independently.
- **Exact inference:** each small component is exhaustively enumerated with backtracking and pruning.
- **Global mine-count constraint:** component solutions are combined with the remaining mine budget and the binomial term
  `C(interior_cells, remaining_mines)`.
- **Marginal probabilities:** `P(cell is a mine)` is calculated for every hidden cell.
- **Interior-cell inference:** cells not touching any revealed number share a probability derived from the expected leftover mine count.
- **Approximate inference:** an MCMC/Gibbs-style constrained sampler is provided for frontier components that exceed the exact-enumeration cap.
- **Sequential decision making:** a pure lowest-risk policy is compared with a risk + information utility policy.
- **Baseline:** a deterministic clue-saturation + subset-difference solver with random guessing when no certain move exists.
- **Experiments:** repeatable simulation on 9x9/10 mines, 16x16/40 mines, and 30x16/99 mines.
- **Visuals:** posterior probability heatmaps and a variable/factor graph.

## Project structure

```text
minesweeper_probabilistic/
├── src/minesweeper_probabilistic/
│   ├── board.py
│   ├── constraints.py
│   ├── baseline.py
│   ├── exact_solver.py
│   ├── mcmc_solver.py
│   ├── policy.py
│   ├── agent.py
│   ├── experiment.py
│   ├── visualize.py
│   └── main.py
├── tests/test_solver.py
├── docs/PROJECT_REPORT.md
├── results/
├── requirements.txt
└── README.md
```

## Mathematical model

For each hidden cell `i`, define a binary random variable:

`X_i ∈ {0,1}` where 1 means mine.

A revealed clue `c` induces a factor/constraint:

`Σ_{i ∈ N(c) ∩ H} X_i = r_c`

where `r_c` is the number still required after subtracting confirmed flagged mines.

For a frontier component `j`, exact enumeration gives the number of valid assignments using `k` mines, `A_j(k)`. If an individual cell `i` is a mine in `A_j^i(k)` of those assignments, then global weighting is performed over all component mine counts and the interior cells.

For a combination with `K` frontier mines, the interior must contain `M-K` mines, so the multiplicity of interior placements is:

`C(I, M-K)`.

Therefore every valid full-board assignment is counted equally under a uniform prior over mine layouts.

## Run it

### 1. Install

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

For a permanent package install (recommended):

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```


### 2. Generate the visual demo

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m minesweeper_probabilistic.main demo
```

Linux/macOS:

```bash
python -m minesweeper_probabilistic.main demo
```


Outputs:

- `results/probability_heatmap.png`
- `results/factor_graph.png`

### 3. Run solver experiments

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m minesweeper_probabilistic.main experiment --games 100
```

Linux/macOS:

```bash
python -m minesweeper_probabilistic.main experiment --games 100
```

Use `--games 300` for the full 900-game comparison (300 games for each board configuration). The same board seed is evaluated by every solver.

Outputs:

- `results/experiment_results.csv`
- `results/win_rate_comparison.png`
- `results/runtime_comparison.png`


### 4. Summarize results

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m minesweeper_probabilistic.main summary
```

Linux/macOS:

```bash
python -m minesweeper_probabilistic.main summary
```


### 5. Play manually

```powershell
.\.venv\Scripts\python.exe -m minesweeper_probabilistic.main game --rows 9 --cols 9 --mines 10
```


## Exact vs approximate inference

Exact inference is used whenever every frontier connected component is no larger than the configured cap (default 64 variables). When a component exceeds the cap, the agent switches to the MCMC solver.

This makes the implementation practical while preserving the exact probabilistic method for small and medium uncertainty regions.

## What to discuss in the presentation

1. A rule-based solver stops whenever no certain deduction exists.
2. The probabilistic solver keeps all valid configurations consistent with the clues.
3. The global mine budget changes the posterior probability of cells, especially isolated interior cells.
4. The best move is a decision under uncertainty: minimum risk is one policy; risk + information is another.
5. On larger boards, exact enumeration becomes exponential in frontier size, motivating MCMC/approximate inference.

## Important scientific note

The experiment script reports measured results from the implementation. Do not claim that the probabilistic solver beats the baseline until the experiment has actually been run and the generated CSV supports that conclusion.
