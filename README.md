# Leakage-aware classification of CO₂-based indoor air quality from low-cost office sensors

Code, configuration and results for the article
**"[Final article title]"**, submitted to the *Journal of Soft Computing and Data Mining* (2026).

The study tests whether the CO₂-based indoor air quality (IAQ) status of an office room
(Poor if CO₂ > 1000 ppm, the ICOP-IAQ 2010 limit) can be classified from **temperature, relative humidity and light only**,
and how much the reported performance depends on the validation protocol.

## Contents

| Path | What it is |
|---|---|
| `config.yaml` | **Every** setting used in the paper: data files, CO₂ threshold, features, seed, validation protocols, model hyperparameter grids, bootstrap and ablation settings |
| `data/` | Occupancy Detection dataset (Candanedo & Feldheim, 2016), unchanged, CC BY 4.0 |
| `src/common.py` | Data loading, feature construction, nested tuning, metrics, day-block bootstrap |
| `src/run_main.py` | 12 classifiers × 3 validation protocols |
| `src/run_ablation.py` | Feature ablation and CO₂-threshold sensitivity (three best models) |
| `src/make_figures.py`, `src/make_tables.py` | Figures and tables in the paper |
| `results/` | Raw outputs: summary metrics, per-day metrics, selected hyperparameters per fold, leave-one-day-out predictions, logs |
| `figures/`, `tables/` | Exactly the figures and tables used in the paper |

## Reproduce

```bash
pip install -r requirements.txt     # Python 3.13; exact versions pinned
./run_all.sh                        # under 1 h on a 2-core CPU
```

All randomness is controlled by `seed: 42` in `config.yaml`. Standardisation and hyperparameter
tuning are performed **inside** each training fold only (nested validation), so no test-day information
reaches model selection.

## Validation protocols

* **A – Random 70/30 split** (stratified; inner 5-fold stratified CV). Included to show the optimism of random splitting.
* **B – Leave-one-day-out** (17 outer folds, one per calendar day; inner 3-fold group CV by day). Main protocol.
* **C – Chronological hold-out**: train 2–10 Feb 2015, test 11–18 Feb 2015 (inner 3-fold group CV by day).

## Main results

F1-score of the Poor-IAQ class (CO₂ > 1000 ppm), inputs: temperature, RH, light.

| Model | A: Random F1 | C: Chronological F1 | B: Leave-one-day-out F1 (95% CI) | B: MCC |
|---|---|---|---|---|
| LR | 0.51 | 0.30 | 0.61 (0.30–0.78) | 0.54 |
| SVM | 0.76 | 0.32 | 0.53 (0.14–0.72) | 0.46 |
| GB | 0.89 | 0.36 | 0.48 (0.12–0.69) | 0.40 |
| NB | 0.57 | 0.30 | 0.48 (0.25–0.65) | 0.38 |
| DT | 0.95 | 0.23 | 0.45 (0.07–0.67) | 0.36 |
| KNN | 0.96 | 0.39 | 0.44 (0.15–0.62) | 0.35 |
| HGB | 0.97 | 0.43 | 0.44 (0.12–0.64) | 0.35 |
| AdaBoost | 0.78 | 0.32 | 0.41 (0.10–0.60) | 0.35 |
| RF | 0.97 | 0.25 | 0.40 (0.15–0.58) | 0.33 |
| LDA | 0.53 | 0.30 | 0.40 (0.17–0.56) | 0.32 |
| MLP | 0.85 | 0.37 | 0.37 (0.07–0.57) | 0.29 |
| ET | 0.97 | 0.36 | 0.36 (0.13–0.51) | 0.28 |

Random splitting inflates scores and reverses the model ranking; see the paper for per-day, occupancy-stratified and ablation results.

## Citation

Please cite the article (see `CITATION.cff`) and the dataset:
Candanedo, L. M., & Feldheim, V. (2016). *Energy and Buildings, 112*, 28–39. https://doi.org/10.1016/j.enbuild.2015.11.071

## Licence

Code: MIT. Data: CC BY 4.0 (original authors).
