# Pilot benchmark (5 games/configuration)

These are measured outputs from this implementation, using the same 5 board seeds across solvers. They are a **pilot**, not the final few-hundred-game experiment.

| Board | Solver | Win rate | Avg moves | Avg time (s) |
|---|---|---:|---:|---:|
| 9x9 / 10 | deterministic | 40% | 10.8 | 0.0032 |
| 9x9 / 10 | probabilistic safety | 100% | 16.8 | 0.0111 |
| 9x9 / 10 | probabilistic utility | 100% | 15.4 | 0.0133 |
| 16x16 / 40 | deterministic | 40% | 51.4 | 0.0293 |
| 16x16 / 40 | probabilistic safety | 40% | 30.4 | 0.0886 |
| 16x16 / 40 | probabilistic utility | 60% | 46.6 | 1.4208 |
| 30x16 / 99 | deterministic | 0% | 11.4 | 0.0056 |
| 30x16 / 99 | probabilistic safety | 80% | 214.2 | 1.7415 |
| 30x16 / 99 | probabilistic utility | 20% | 75.8 | 0.8576 |

Do not generalize these percentages statistically. Run the requested 300-game experiment before making a journal-style claim about win-rate superiority.
