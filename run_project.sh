#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 -m minesweeper_probabilistic.main demo
PYTHONPATH=src python3 -m minesweeper_probabilistic.main experiment --games 300
PYTHONPATH=src python3 -m minesweeper_probabilistic.main summary
