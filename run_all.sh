#!/usr/bin/env bash
# Reproduce every result and figure in the paper. Total run time under 1 h on a 2-core CPU.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src PYTHONWARNINGS=ignore
python src/run_main.py random,chronological      # Protocols A and C
python src/run_main.py leave_one_day_out         # Protocol B (17 folds, nested tuning)
python src/run_ablation.py                       # feature and CO2-threshold ablations
python src/make_figures.py                       # figures/*.png
python src/make_tables.py                        # tables/*.csv used in the paper
