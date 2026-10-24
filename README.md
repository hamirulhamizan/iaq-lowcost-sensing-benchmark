# Leakage-aware classification of indoor air quality from low-cost sensors

Code, configuration and results for the article
**"Can Low-Cost Ambient Sensors Screen Indoor Air Quality? A Leakage-Aware Benchmark of Twelve Machine Learning Classifiers on CO₂-Based and Multi-Pollutant Labels"**, submitted to the *Journal of Soft Computing and Data Mining* (2026).

The study tests whether indoor air quality (IAQ) status can be classified from **low-cost temperature, relative humidity
and light or sound sensors only**, and how much the reported performance depends on the validation protocol.

* **Study 1** – office room (Candanedo & Feldheim, 2016): CO₂-based label (CO₂ > 1000 ppm, ICOP-IAQ 2010).
* **Study 2** – 95 home offices (Rysanek, 2024): multi-pollutant label (CO₂ > 1000 ppm, PM2.5 > 15 µg/m³ WHO 2021, TVOC > 3 ppm ICOP-IAQ 2010); validation that holds out whole homes.
* **Supplementary Study 3** – high-pollution datasets from India (DALTON) and Croatia (Lovrić et al., 2025): label prevalence under WHO vs Malaysian (MAAQS 2020) limits and a sensitivity analysis.

## Contents

| Path | What it is |
|---|---|
| `config.yaml` | **Every** setting used in the paper: data files, CO₂ threshold, features, seed, validation protocols, model hyperparameter grids, bootstrap and ablation settings |
| `data/` | Occupancy Detection (CC BY 4.0), work-from-home (CC BY 4.0) and Lovrić et al. (CC BY 4.0) data; DALTON is not redistributed (AGPL, non-commercial) |
| `src/common.py` | Data loading, feature construction, model pipelines, metrics |
| `src/engine.py` | Identical tuning for every model: PR-AUC grid search, inner-fold F1-optimal threshold, block bootstrap |
| `src/runner.py` | Checkpointing of every run (resumable) |
| `src/run_stats.py` | Friedman, Nemenyi and Holm-corrected Wilcoxon tests |
| `src/run_episodes.py` | Episode-level detection, delay and false-alarm episodes |
| `src/run_main.py` | 12 classifiers × 3 validation protocols |
| `src/run_ablation.py` | Feature ablation and CO₂-threshold sensitivity (three best models) |
| `src/run_multisite.py` | Study 2: 12 classifiers × random / leave-sites-out / chronological, composite and single-pollutant labels |
| `src/run_multisite_ablation.py` | Study 2: feature ablation and PM2.5-limit sensitivity (15 / 25 / 35 µg/m³) for the three best models |
| `src/run_sensitivity.py` | Supplementary Study 3: DALTON and Lovrić under WHO and Malaysian limits |
| `src/data_prep/` | Download, verification and preparation scripts for the Study 2–3 datasets (see its README) |
| `src/make_figures.py`, `src/make_tables.py`, `src/make_multisite_outputs.py` | Figures and tables in the paper |
| `results/` | Raw outputs: summary metrics, per-day metrics, selected hyperparameters per fold, leave-one-day-out predictions, logs |
| `figures/`, `tables/` | Exactly the figures and tables used in the paper |

## Reproduce

```bash
pip install -r requirements.txt     # Python 3.13; exact versions pinned
./run_all.sh                        # about 6 h on a 2-core CPU; resumes after interruption
```

All randomness is controlled by `seed: 42` in `config.yaml`. Standardisation and hyperparameter
tuning are performed **inside** each training fold only (nested validation), so no test-day information
reaches model selection.

## Validation protocols

* **A – Random 70/30 split** (stratified; inner 3-fold stratified CV). Included to show the optimism of random splitting.
* **B – Leave-one-day-out** (Study 1; 17 outer folds; inner 3-fold group CV by day) or **leave-sites-out** (Study 2; 5 outer folds over 95 homes; inner 3-fold group CV by home). Main protocol.
* **C – Chronological hold-out**: Study 1 train 2–10 Feb, test 11–18 Feb 2015; Study 2 first 70% vs last 30% of the period.

## Main results

### Study 1 – F1-score of the Poor-IAQ class (CO₂ > 1000 ppm), inputs: temperature, RH, light

Every model is tuned identically: grid search on PR-AUC, then an F1-optimal decision threshold chosen on inner out-of-fold scores (training data only).

| Model | A: Random F1 | C: Chronological F1 | B: Leave-one-day-out F1 (95% CI) | B: MCC | B: PR-AUC | B: F1 at default threshold |
|---|---|---|---|---|---|---|
| DT | 0.91 | 0.23 | 0.58 (0.28–0.76) | 0.51 | 0.57 | 0.42 |
| AdaBoost | 0.82 | 0.31 | 0.57 (0.27–0.74) | 0.50 | 0.57 | 0.02 |
| RF | 0.95 | 0.30 | 0.53 (0.24–0.71) | 0.46 | 0.59 | 0.37 |
| GB | 0.89 | 0.28 | 0.52 (0.23–0.71) | 0.44 | 0.57 | 0.44 |
| LR | 0.56 | 0.27 | 0.52 (0.22–0.75) | 0.43 | 0.73 | 0.36 |
| LDA | 0.56 | 0.30 | 0.52 (0.24–0.71) | 0.42 | 0.72 | 0.40 |
| HGB | 0.97 | 0.36 | 0.50 (0.24–0.66) | 0.42 | 0.57 | 0.41 |
| KNN | 0.96 | 0.36 | 0.48 (0.24–0.64) | 0.38 | 0.48 | 0.41 |
| ET | 0.90 | 0.32 | 0.48 (0.26–0.63) | 0.38 | 0.68 | 0.00 |
| SVM | 0.85 | 0.36 | 0.48 (0.19–0.71) | 0.38 | 0.63 | 0.39 |
| MLP | 0.85 | 0.36 | 0.44 (0.21–0.62) | 0.33 | 0.67 | 0.40 |
| NB | 0.57 | 0.30 | 0.39 (0.23–0.49) | 0.27 | 0.74 | 0.48 |

Friedman test over the 10 days with Poor IAQ: p = 0.63 (no significant difference between models). Random splitting inflates scores; see the paper for per-day, occupancy-stratified, episode-level and ablation results.

### Study 2 – multi-pollutant label in 95 home offices, inputs: temperature, RH, sound level

| Model | Random F1 | Chronological F1 | Leave-sites-out F1 (95% CI) | LSO MCC | LSO F1, CO₂ only | LSO F1, PM2.5 only |
|---|---|---|---|---|---|---|
| LR | 0.42 | 0.33 | 0.41 (0.33–0.48) | 0.29 | 0.34 | 0.13 |
| LDA | 0.42 | 0.33 | 0.41 (0.33–0.48) | 0.29 | 0.34 | 0.15 |
| GB | 0.46 | 0.32 | 0.40 (0.32–0.48) | 0.29 | 0.35 | 0.12 |
| ET | 0.47 | 0.37 | 0.40 (0.32–0.47) | 0.29 | 0.35 | 0.11 |
| MLP | 0.45 | 0.34 | 0.40 (0.32–0.48) | 0.28 | 0.34 | 0.08 |
| HGB | 0.48 | 0.33 | 0.40 (0.32–0.47) | 0.28 | 0.34 | 0.10 |
| AdaBoost | 0.44 | 0.31 | 0.40 (0.32–0.47) | 0.28 | 0.34 | 0.09 |
| RF | 0.48 | 0.32 | 0.39 (0.32–0.45) | 0.27 | 0.34 | 0.12 |
| KNN | 0.49 | 0.32 | 0.38 (0.31–0.44) | 0.25 | 0.31 | 0.12 |
| SVM | 0.42 | 0.27 | 0.37 (0.29–0.45) | 0.25 | 0.31 | 0.10 |
| NB | 0.38 | 0.33 | 0.37 (0.29–0.45) | 0.24 | 0.32 | 0.12 |
| DT | 0.45 | 0.31 | 0.37 (0.30–0.44) | 0.24 | 0.34 | 0.10 |

## Citation

Please cite the article (see `CITATION.cff`) and the dataset:
Candanedo, L. M., & Feldheim, V. (2016). *Energy and Buildings, 112*, 28–39. https://doi.org/10.1016/j.enbuild.2015.11.071

## Licence

Code: MIT. Data: CC BY 4.0 (original authors) for the three included datasets; DALTON is not included (AGPL v3, non-commercial research) and is rebuilt by the scripts in `src/data_prep/`.
