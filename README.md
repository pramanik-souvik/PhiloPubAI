# When Accuracy Is Not Knowledge — Empirical AI Implementation-Readiness Audit (Dhaka)

Code, data and results for the paper

> **When Accuracy Is Not Knowledge: An Empirical Implementation-Readiness Audit of Explainable AI for Occupational Lung-Disease Surveillance in Dhaka**
> Tithi Mitra, Souvik Pramanik, Most. Aysha Siddika Sumona, Ahmed Faizul Haque Dhrubo, Farhana Sharmin, Mohammad Abdul Qayum

This study is the empirical phase of the authors' earlier problem-discovery study, *From Knowledge to Legitimacy: A Philosophical Problem Discovery of AI Implementation Readiness in Public Health Disease Surveillance*. That paper proposed four readiness dimensions. Here they become 18 measurable indicators, which are applied to an explainable-AI use case:

| Dimension | What is measured |
|---|---|
| **E** Epistemic adequacy | label–source association, provenance share of discrimination, source recoverability, labelling artefacts, ambient-exposure resolution |
| **J** Distributive justice | exposure burden vs. explicit health coding, ambient-data coverage, error-rate (TPR/FPR) gaps across occupations |
| **G** Ethical governance | SHAP stability, cross-method agreement, expert–data concordance, counterfactual actionability, re-identification risk |
| **L** Institutional legitimacy | pooled and within-occupation AUC, leave-one-occupation-out transfer, calibration, decision-curve net benefit |

## Repository structure

```
.
├── Readiness_Analysis_Pipeline.ipynb   # the full analysis in a single cell
├── readiness_pipeline.py               # identical code as a plain script
├── requirements.txt
├── data/
│   ├── README.md                       # codebook and coding notes
│   ├── survey_harmonised.csv           # harmonised survey, 7,599 respondents
│   └── area_pm25_lookup.csv            # CAMS reanalysis PM2.5 (2024 mean) by zone
└── results/                            # outputs of the full run reported in the paper
    ├── numbers.tex                     # every value reported in the paper
    ├── tab_*.tex                       # tables (LaTeX, booktabs)
    ├── fig_*.pdf / fig_*.png           # figures
    ├── csv/                            # machine-readable tables + per-fold CV scores
    └── run_info.json                   # configuration, package versions, runtime
```

## Reproducing the results

```bash
pip install -r requirements.txt
jupyter notebook Readiness_Analysis_Pipeline.ipynb   # run the single cell
```

Or, without Jupyter:

```bash
python readiness_pipeline.py
```

Run from the repository root. A full run (5×5 repeated stratified cross-validation) takes about 11–16 minutes on a laptop and rewrites `results/`. Set `FAST=True` in the `CFG` block at the top for a ~2-minute smoke test; its numbers are not paper-grade. All randomness is seeded (`SEED=42`).

## Analysis summary

- **Outcome coding.** Blank disease fields mean *no disease*, as confirmed by the data collectors. This is the primary coding (Scenario B). Excluding blanks, as the original pipeline did, is reported as a contrast (Scenario A).
- **Models.** Logistic regression, random forest and XGBoost are compared against a prior-only baseline. Preprocessing is fitted inside each training fold, and five feature sets are compared: occupation, demographics, exposure, exposure + demographics, and full.
- **Validation.** Repeated cross-validation with Nadeau–Bengio corrected *t*-tests, leave-one-occupation-out validation, and models fitted within each occupation.
- **Adjusted models.** Logistic GLMs with cluster-robust standard errors, clustered by residential zone.
- **Main finding.** Pooled performance is high: AUC ≈ 0.90, well calibrated, with positive net benefit. Within occupations, and when transferred to an unseen occupation, performance is at chance. Most of the apparent exposure signal reflects which survey sheet a record came from. Only 5 of 18 readiness criteria are met (`results/tab_scorecard.tex`).

## Citation

If you use this code or data, please cite the accompanying paper (details above).
