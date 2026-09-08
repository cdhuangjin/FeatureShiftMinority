# Project C Gate A

**中文工作题目**：类别不平衡表格分类中特征可用性变化对少数类鲁棒性的影响

**English**：Minority-Class Vulnerability under Feature Availability Shift in Imbalanced Tabular Classification

This repository contains two distinct research stages for Project C:

> - **Gate A** = exploratory diagnostic
> - **Phase C1/C2** = paper-level validation

They share the same core question but differ in rigor: Gate A is a small, CPU-rapid
first-pass diagnostic; Phase C1/C2 is the statistically robust, paper-oriented follow-up.

## Gate A (exploratory diagnostic)

Gate A validates one question:

> When a subset of features becomes unavailable at deployment time, does the minority class lose proportionally more performance than the majority class, and is that loss invisible to overall `AUROC`/`AUPRC`?

The project is deliberately tiny and CPU-rapid:

- 3 imbalanced public binary tabular datasets
- 2 models (XGBoost, Random Forest)
- 3 seeds
- 4 feature-availability environments (`full`, `random`, `global-important`, `minority-specific` removal)
- 3 feature-importance views (global, minority, majority) plus the Minority Specificity Index (MSI)

## Layout

```text
FeatureShiftMinority/
├── README.md
├── requirements.txt
├── configs/gate_a.yaml
├── data/{raw,processed}
├── src/{data,preprocessing,models,feature_importance,feature_shift,metrics,vulnerability,gate_decision,plotting}.py
├── scripts/reproduce_gate_a.py
├── tests/test_*.py
└── results/gate_a/
```

## Run

```bash
python scripts/reproduce_gate_a.py
```

The script performs: data load → stratified split → preprocessing → training → importance analysis → feature masking → evaluation → vulnerability computation → figures → automatic gate decision → report and manifest.

## Gate result

The automatic gate decision and all evidence tables/figures are written under `results/gate_a/`. See `results/gate_a/gate_report.md` for the summary.

## Phase C1/C2 (paper-level validation)

Phase C1/C2 is the scaled, peer-review-grade follow-up. It answers the same question
but with cross-dataset, cross-model, cross-mechanism, cross-severity statistical evidence.

- 10 imbalanced public binary tabular datasets (6 observable, 4 floor-risk)
- 4 models (Logistic Regression, Random Forest, XGBoost, LightGBM)
- 3 seeds
- removal-severity curves (0/5/10/20/30/40 %)
- 4 removal mechanisms (`random`, `global_importance`, `minority_specific`, `majority_specific`)
- structured feature loss (group / correlated / matched-random), delayed availability
- baseline-observability screen and floor-effect handling
- bootstrap CI, Wilcoxon signed-rank, BH FDR, rank-biserial effect size
- an automatic Paper Gate (`STRONG-GO` / `GO` / `HOLD` / `STOP-PIVOT`)

All Phase C1/C2 artefacts are written under `results/phase_c12/` (never
`results/gate_a/`). The entry point is:

```bash
python scripts/reproduce_phase_c12.py --stage all
```

Key outputs: `results/phase_c12/phase_c12_report.md`, `paper_gate.json`,
`manifest.json`, `dataset_profiles.csv`, `raw_results.csv`, `vulnerability_results.csv`,
`slope_results.csv`, `statistics_results.csv`, `class_specific_control.csv`, and the
figure suite under `results/phase_c12/figures/`.

## Stage roles

| Stage | Purpose | Output |
|---|---|---|
| Gate A | Exploratory diagnostic | `results/gate_a/` |
| Phase C1/C2 | Paper-level validation | `results/phase_c12/` |

`results/gate_a/` is treated as immutable: Phase C1/C2 only ever writes to
`results/phase_c12/`.
