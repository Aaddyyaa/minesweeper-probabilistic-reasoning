# Minesweeper as a Probabilistic Reasoning Problem

## 1. Abstract

Minesweeper is commonly presented as a deterministic logic puzzle, but many board states do not admit a certain move. At those states a solver must reason about multiple possible hidden mine layouts and choose an action under uncertainty. This project models each hidden cell as a binary random variable, expresses revealed numbers as sum constraints, and performs posterior inference over valid mine configurations. Exact inference enumerates independent frontier components and combines them with the global mine-count constraint. A Markov-chain Monte Carlo fallback is supplied for large frontiers. The resulting posterior probabilities are used by decision policies that either minimize mine risk or trade a small amount of risk for expected information gain. The system is evaluated against a deterministic rule-based baseline on three board sizes.

## 2. Objectives

- Formulate Minesweeper as a probabilistic graphical model.
- Implement exact marginal inference on small connected frontier components.
- Correctly incorporate the fixed total number of mines on the board.
- Estimate posterior mine probabilities for frontier and interior cells.
- Compare probabilistic decision making with deterministic rules.
- Explore approximate inference when exact enumeration becomes expensive.

## 3. Graphical-model formulation

Let `X_i` be a binary variable for every hidden cell. Each revealed clue creates a factor connecting the neighboring hidden cells. The factor is nonzero only when the sum of its neighboring variables equals the clue's remaining mine count.

This is an undirected factor graph / Markov-network representation. Cells in separate frontier components have no shared factors and are conditionally independent given the current revealed state and the global mine count. The implementation therefore decomposes the frontier before enumeration.

## 4. Exact inference

For each connected component, the solver enumerates assignments using depth-first backtracking. After every variable assignment it checks each touched constraint using:

`assigned_mines <= target <= assigned_mines + unassigned_variables`

This pruning removes large portions of the binary search tree.

For each component the implementation stores:

- number of valid assignments for each local mine count `k`;
- number of assignments in which each individual cell is a mine, grouped by `k`.

The global posterior is formed by combining the components. If their assignments contain `K` frontier mines and `M` mines remain, the interior must contain `M-K` mines. With `I` interior cells there are `C(I, M-K)` equally likely placements. This is the crucial global correction that purely local rule systems miss.

## 5. Interior cells

Interior cells have no currently active clue factor. They are not assigned an arbitrary probability. Their posterior is derived from the expected number of leftover mines after accounting for all consistent frontier configurations. Because the interior cells are symmetric under the model, they share the same marginal probability.

## 6. Decision policies

### Safety policy

- Click any cell with posterior `P(mine)=0`.
- Flag any cell with posterior `P(mine)=1`.
- Otherwise click the minimum-probability cell.

### Risk + information utility policy

With win utility `+1` and loss utility `-1`, the expected terminal utility of clicking cell `i` is:

`EU_i = (1-p_i)(+1) + p_i(-1) = 1 - 2p_i`.

The implementation adds a small information bonus based on the number of unrevealed neighbors and nearby revealed clues. This is intentionally a transparent heuristic rather than a claim of globally optimal information gain.

## 7. Baseline solver

The baseline uses deterministic clue saturation and subset-difference reasoning. In practice this covers the common 1-1 / 1-2 pattern family and more general deductions that can be expressed as set differences. When no deterministic safe move exists, it selects a random hidden cell.

## 8. Approximate inference

When a frontier component exceeds the exact-enumeration cap, the probabilistic agent switches to a constrained MCMC sampler. The sampler maintains the global mine count and proposes mine/safe swaps; a swap is accepted only when all active clue constraints remain satisfied. Marginal probabilities are estimated from the retained samples.

This makes the computation scalable, while clearly separating exact results from approximate results in the recorded experiment metadata.

## 9. Experimental design

Board configurations:

| Configuration | Rows | Columns | Mines |
|---|---:|---:|---:|
| Easy | 9 | 9 | 10 |
| Medium | 16 | 16 | 40 |
| Hard | 30 | 16 | 99 |

Each solver receives reproducible board seeds. The same board seed is used across the competing solvers for fair structural comparison. Metrics include win/loss, number of clicks, computation time, maximum frontier size, exact-vs-MCMC inference calls, and number of chosen 50/50 moves.

## 10. Results

The repository stores measured experiment output in `results/experiment_results.csv`. The benchmark is deliberately generated by the program rather than hard-coded into the report. Use the following commands for the final study:

```powershell
.\\.venv\\Scripts\\python.exe -m minesweeper_probabilistic.main experiment --games 300
.\\.venv\\Scripts\\python.exe -m minesweeper_probabilistic.main summary
```

The benchmark contains 900 games in total: 300 each for the easy, medium, and hard configurations, with the same board seed assigned to all three policies. Report the generated CSV values directly; do not replace them with pilot or fabricated numbers.

## 11. Expected findings

The probabilistic solver should have an advantage whenever the board reaches states with multiple consistent mine layouts. The advantage is expected to become more valuable as the board becomes larger and local certainty becomes rarer. The runtime cost also grows, because exact inference is exponential in frontier component size. The heatmap is therefore a central result: it makes uncertainty visible cell-by-cell and shows that an interior cell can have a posterior substantially different from a nearby frontier guess.

## 12. Failure modes and limitations

- Exact enumeration becomes exponential for large tightly connected frontiers.
- The MCMC fallback is approximate and should be evaluated for convergence.
- The information utility is heuristic rather than a full value-of-information calculation.
- A random board generator with only first-click safety can still produce difficult or forced-gamble situations.
- The baseline is intentionally simple and should not be compared against a highly optimized commercial Minesweeper engine without a separate benchmark definition.

## 13. Syllabus mapping

| Unit | Project component |
|---|---|
| Unit 1 | Markov network, factor graph, frontier components, conditional independence |
| Unit 2 | Exact inference, enumeration, variable elimination interpretation |
| Unit 3 | MCMC / approximate posterior estimation |
| Units 5-6 | Expected utility, risk-sensitive sequential decisions, information-aware action selection |

## 14. Real-world analogies

- **Robot mapping:** infer occupancy from local sensor constraints.
- **Group testing:** infer which items are positive from pooled counts.
- **Fault diagnosis:** infer failed components from observed symptom constraints.
- **Risk-aware exploration:** select the next action by balancing safety and information gain.
