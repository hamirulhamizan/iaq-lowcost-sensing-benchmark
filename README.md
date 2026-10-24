# Leakage-aware classification of indoor air quality from low-cost sensors

Code, configuration and results for the article
**"[Final article title]"**, submitted to the *Journal of Soft Computing and Data Mining* (2026).

The study tests whether indoor air quality (IAQ) status can be classified from **low-cost temperature, relative humidity
and light or sound sensors only**, and how much the reported performance depends on the validation protocol.

* **Study 1** – office room (Candanedo & Feldheim, 2016): CO₂-based label (CO₂ > 1000 ppm, ICOP-IAQ 2010).
* **Study 2** – 95 home offices (Rysanek, 2024): multi-pollutant label (CO₂ > 1000 ppm, PM2.5 > 15 µg/m³ WHO 2021, TVOC > 3 ppm ICOP-IAQ 2010); validation that holds out whole homes.
* **Study 3** – high-pollution datasets from India (DALTON) and Croatia (Lovrić et al., 2025): label prevalence under WHO vs Malaysian (MAAQS 2020) limits and a sensitivity analysis.

## Contents

| Path | What it is |
|---|---|
| `config.yaml` | **Every** setting used in the paper: data files, CO₂ threshold, features, seed, validation protocols, model hyperparameter grids, bootstrap and ablation settings |
| `data/` | Occupancy Detection (CC BY 4.0), work-from-home (CC BY 4.0) and Lovrić et al. (CC BY 4.0) data; DALTON is not redistributed (AGPL, non-commercial) |
| `src/common.py` | Data loading, feature construction, nested tuning, metrics, day-block bootstrap |
| `src/run_main.py` | 12 classifiers × 3 validation protocols |
| `src/run_ablation.py` | Feature ablation and CO₂-threshold sensitivity (three best models) |
| `src/run_multisite.py` | Study 2: 12 classifiers × random / leave-sites-out / chronological, composite and single-pollutant labels |
| `src/run_sensitivity.py` | Study 3: DALTON and Lovrić under WHO and Malaysian limits |
| `src/data_prep/` | Download, verification and preparation scripts for the Study 2–3 datasets (see its README) |
| `src/make_figures.py`, `src/make_tables.py` | Figures and tables in the paper |
| `results/` | Raw outputs: summary metrics, per-day metrics, selected hyperparameters per fold, leave-one-day-out predictions, logs |
| `figures/`, `tables/` | Exactly the figures and tables used in the paper |

## Reproduce

```bash
pip install -r requirements.txt     # Python 3.13; exact versions pinned
./run_all.sh                        # about 3 h on a 2-core CPU
```

All randomness is controlled by `seed: 42` in `config.yaml`. Standardisation and hyperparameter
tuning are performed **inside** each training fold only (nested validation), so no test-day information
reaches model selection.

## Validation protocols

* **A – Random 70/30 split** (stratified; inner 5-fold stratified CV). Included to show the optimism of random splitting.
* **B – Leave-one-day-out** (17 outer folds, one per calendar day; inner 3-fold group CV by day). Main protocol.
* **C – Chronological hold-out**: train 2–10 Feb 2015, test 11–18 Feb 2015 (inner 3-fold group CV by day).

## Main results

### Study 1 – F1-score of the Poor-IAQ class (CO₂ > 1000 ppm), inputs: temperature, RH, light.

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

### Study 2 – multi-pollutant label in 95 home offices, inputs: temperature, RH, sound level

| Model | Random F1 | Chronological F1 | Leave-sites-out F1 (95% CI) | LSO F1, CO₂ only | LSO F1, PM2.5 only |
|---|---|---|---|---|---|
| LR | 0.41 | 0.30 | 0.40 (0.32–0.47) | 0.34 | 0.12 |
| DT | 0.37 | 0.29 | 0.26 (0.21–0.30) | 0.22 | 0.05 |
| KNN | 0.42 | 0.35 | 0.25 (0.20–0.30) | 0.19 | 0.03 |
| RF | 0.36 | 0.38 | 0.22 (0.16–0.28) | 0.13 | 0.01 |
| HGB | 0.33 | 0.39 | 0.22 (0.15–0.27) | 0.13 | 0.01 |
| MLP | 0.25 | 0.39 | 0.21 (0.14–0.26) | 0.12 | 0.00 |
| GB | 0.28 | 0.37 | 0.20 (0.14–0.25) | 0.12 | 0.01 |
| LDA | 0.20 | 0.40 | 0.19 (0.12–0.26) | 0.13 | 0.00 |
| AdaBoost | 0.15 | 0.35 | 0.14 (0.09–0.18) | 0.06 | 0.00 |
| SVM | 0.19 | 0.34 | 0.13 (0.08–0.17) | 0.05 | 0.00 |
| NB | 0.12 | 0.13 | 0.08 (0.03–0.16) | 0.00 | 0.00 |
| ET | 0.01 | 0.02 | 0.01 (0.00–0.01) | 0.00 | 0.00 |

## Citation

Please cite the article (see `CITATION.cff`) and the dataset:
Candanedo, L. M., & Feldheim, V. (2016). *Energy and Buildings, 112*, 28–39. https://doi.org/10.1016/j.enbuild.2015.11.071

## Licence

Code: MIT. Data: CC BY 4.0 (original authors) for the three included datasets; DALTON is not included (AGPL v3, non-commercial research) and is rebuilt by the scripts in `src/data_prep/`.
