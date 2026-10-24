#!/usr/bin/env bash
# Reproduce every result, table and figure in the paper (about 3 h on a 2-core CPU, 7 GB RAM).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src PYTHONWARNINGS=ignore
# Study 1: office room, CO2-based label
python src/run_main.py random,chronological
python src/run_main.py leave_one_day_out
python src/run_ablation.py
# Study 2: 95 home offices, multi-pollutant label
python src/run_multisite.py random,leave_sites_out,chronological composite
python src/run_multisite.py leave_sites_out CO2
python src/run_multisite.py leave_sites_out PM2.5
python src/run_multisite_ablation.py   # Study 2 feature ablation and PM2.5-limit sensitivity
# Study 3: high-pollution datasets (requires data/dalton/dalton_hourly.csv, see src/data_prep/README.md)
python src/run_sensitivity.py
# Figures and tables
python src/make_tables.py
python src/make_multisite_outputs.py
python src/make_figures.py
