# Data preparation (Studies 2 and 3)

| Dataset | Licence | In this repository? | How to rebuild |
|---|---|---|---|
| Work-from-home IEQ (Rysanek, 2024), https://doi.org/10.17632/7ggv5vhpwb.1 | CC BY 4.0 | yes, `data/wfh/wfh_long.csv` | download `IEQ-WFH-monitoring-repo_v1.1.xlsx`, then `python src/data_prep/wfh_reshape.py IEQ-WFH-monitoring-repo_v1.1.xlsx data/wfh/wfh_long.csv` |
| Lovrić et al. (2025), https://doi.org/10.5281/zenodo.14763137 | CC BY 4.0 | yes, `data/lovric/lovric_minute.csv` | download and unzip `data.zip` (md5 e7c316b97a518770b0386aa507d9f570), then `python src/data_prep/lovric_build.py <unzipped>/data/data data/lovric/lovric_minute.csv` |
| DALTON (Karmakar et al., 2024), https://github.com/prasenjit52282/dalton-dataset | GNU AGPL v3, non-commercial research | **no** (licence) | `git clone` the repository (LFS pointers only), then `python src/data_prep/dalton_fetch.py <clone> <minute_dir>` (downloads every processed file, verifies its SHA-256 against the LFS pointer, averages to 1 min), `python src/data_prep/dalton_clean.py <minute_dir> dalton_minute_clean.csv`, `python src/data_prep/dalton_hourly.py dalton_minute_clean.csv data/dalton/dalton_hourly.csv` |

Cleaning rules for DALTON: zero-filled gap rows removed; minutes with < 50% valid seconds removed; timestamps outside 2022-09-01 – 2024-04-30 removed; site F2 excluded (identical to F1); CO, NO2 and VOC channels not used (uncalibrated metal-oxide signals).
