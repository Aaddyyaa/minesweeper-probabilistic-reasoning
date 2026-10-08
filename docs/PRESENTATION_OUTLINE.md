# 10-slide presentation outline

## Slide 1 — Title
**Minesweeper as a Probabilistic Reasoning Problem**

Names / course / guide / institution.

## Slide 2 — Why Minesweeper is probabilistic
Deterministic rules solve only states with certain moves. In ambiguous states several mine layouts remain possible, so the correct question becomes: `What is P(X_i = mine | clues)?`

## Slide 3 — Markov network
Show the generated `results/factor_graph.png`.
Explain variable nodes, clue factors, and why disconnected frontier components can be solved independently.

## Slide 4 — Exact inference
1. Build frontier.
2. Split into components.
3. Enumerate valid assignments by backtracking.
4. Group assignments by mine count.

## Slide 5 — Global mine-count constraint
For component assignments using `K` frontier mines, the interior contributes
`C(I, M-K)` configurations.
This changes posterior probabilities and is the main correction over naive local guessing.

## Slide 6 — Probability heatmap
Show `results/probability_heatmap.png`.
Point out cells near 0, near 1, and intermediate probabilities.
Explain the interior-cell posterior.

## Slide 7 — Decision policies
Safety: minimum `P(mine)`.
Utility: `EU = 1 - 2p + λ·information`.
Discuss why information can justify a slightly riskier move.

## Slide 8 — Baseline vs probabilistic solver
Use `results/example_win_rates.png` for the pilot illustration.
Make clear that the final reported benchmark should use the requested 300-game run, not the 5-game pilot.

## Slide 9 — Exact vs MCMC scalability
Explain the exponential frontier cost of exact enumeration and the MCMC fallback for larger uncertainty regions.
Use `results/example_runtime.png` and report exact/MCMC call counts.

## Slide 10 — Applications and conclusion
Robot occupancy mapping, pooled medical testing, fault diagnosis, and risk-aware exploration.
Conclusion: Minesweeper is a compact example of graphical-model inference plus sequential decision making.
