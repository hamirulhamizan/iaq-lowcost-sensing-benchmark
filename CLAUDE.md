# Project notes for Claude Code

This repository holds the code, configuration and results for a journal article being revised for the
**Journal of Soft Computing and Data Mining (JSCDM, Penerbit UTHM, Scopus-indexed)**.

Current title: *Indoor Air Quality Exceedance Classification from Low-Cost Ambient Sensors: A Leakage-Aware
Evaluation of Twelve Machine Learning Classifiers and Simple Rules* (manuscript version v7).

## What the study does
- **Study 1** – office room (Occupancy Detection dataset, Candanedo & Feldheim, 2016; 20,560 one-minute records,
  17 days). Label: CO₂ exceedance (> 1000 ppm, ICOP-IAQ 2010). Inputs: temperature, RH, light.
- **Study 2** – 95 home offices (Manu & Rysanek, 2024; 120,590 hourly records). Label: CO₂ > 1000 ppm or
  TVOC > 3 ppm (ICOP-IAQ 2010) or PM2.5 > 15 µg/m³ (WHO 2021, applied to hourly means). Inputs: T, RH, sound level.
- **Supplementary Study 3** – DALTON (India) and Lovrić et al. (Croatia): label prevalence under WHO vs Malaysian
  (MAAQS 2020) limits. DALTON is AGPL and is not redistributed; `src/data_prep/` rebuilds it.
- Pollutant concentrations are never model inputs.

## Method (must stay identical for every model)
- 12 sklearn classifiers (LR, LDA, NB, KNN, DT, RF, ET, AdaBoost, GB, HGB, SVM, MLP); no class weights.
- `src/engine.py`: grid search on PR-AUC (average precision) → F1-optimal threshold on inner out-of-fold scores
  → refit. Optional 10% false-alarm-budget threshold (`alarm_budgets` in config). Kernel SVM capped at 20,000 rows.
- Protocols: A random 70/30; B leave-one-day-out (S1) / leave-sites-out, 5 groups (S2); C chronological;
  D buffered LODO (±1 day removed); E forward chaining with chronological inner folds; F unseen homes + later period.
- Simple-rule baselines under the same outer splits: always alarm, light/RH/sound threshold, working-hours rule.
- Every setting lives in `config.yaml`; seed 42. Runs are checkpointed in `results/<study>/runs/` (gitignored).

## Key results (v7)
- S1: random F1 up to 0.97; leave-one-day-out 0.39–0.58 (DT best); Friedman p = 0.63 (no statistically detectable
  difference); light threshold 0.51; forward chaining 0.04–0.47; buffered 0.29–0.46.
- S2: leave-sites-out 0.37–0.41 (LR best); RH threshold 0.38; PM2.5-only MCC ≤ 0.12; unseen homes + future 0.29–0.33,
  RH threshold 0.35.
- Classifiers are in alarm ~20–34% of the time; most detected episodes were already in alarm at onset.
- Conclusion is deliberately narrow: such signals are at most a potential supplementary indicator of CO₂ exceedances;
  validation design changes conclusions; ML adds little over simple rules on these data.

## Reproduce
`pip install -r requirements.txt && ./run_all.sh` (≈ 8 h on 2 cores; resumable). Output files:
`results/study1|study2/summary*.csv`, `results/temporal_summary.csv`, `results/budget_summary.csv`,
`results/baselines/`, `results/episodes/`, `results/stats/`, `tables/`, `figures/` (Fig. 1 from `src/make_framework.py`).

## Writing rules from the author
- Refer to the research as **"this study"** — never "we", "our" or "us" (also in response letters: "the authors").
- Text and references in **APA 7**; prefer 2024–2026 Scopus/WoS sources; minimise pre-2020 references.
- JSCDM order: title page, abstract (≤ 300 words), 3–5 keywords, text (Introduction, Related Work, Methods, Results,
  Discussion, Conclusion), acknowledgments, references; official JSCDM Word template; ≥ 10 pages.
- Use "exceedance / non-exceedance", not "Good/Poor IAQ"; do not claim practical screening usefulness.
- Every number in the paper must come from the files in `results/` or `tables/`; check after any rerun.

## History
- Reviewer round 1 (v6): unfair imbalance handling (LR-only class weights) → identical tuning; PR-AUC; statistics;
  episode metrics; harmonised settings; re-tuned ablations; Study 3 moved to the supplement.
- Reviewer round 2 (v7): baselines, alarm burden, corrected episode matching, temporal protocols D–F, budget recall,
  paired bootstrap differences, all-day Friedman, exceedance wording, Manu & Rysanek citation, Candanedo framing.

## Still pending
- Author names, affiliations, ORCID, acknowledgement/grant and author-contribution statement (highlighted in the
  manuscript; placeholders `[Surname]` in `CITATION.cff` and `.zenodo.json`).
- After author details are filled in: create a GitHub release → Zenodo DOI (Zenodo–GitHub link is already switched
  on) → put the DOI into the manuscript, README, `CITATION.cff` and cover letter.

## Not in this repository
The manuscript, supplement, response letter and cover letter are Word files built by private scripts (kept out of
this public repo so the unpublished text is not exposed). Ask the author for the "manuscript toolkit" zip if they
need to be rebuilt.
