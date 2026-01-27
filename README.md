# Radiomics-Based Detection and Temporal Characterization of Masseter Muscle Atrophy Using Ultrasound and Machine Learning

This repository contains the complete and reproducible analysis pipeline for a study evaluating radiomics-based machine learning applied to ultrasound images to detect and temporally characterize masseter muscle atrophy induced by botulinum toxin type A (BoNT-A). Quantitative histological measurements are used as a comparative reference.

The project was designed and reported in accordance with the ARRIVE, STARD, and CLAIM guidelines.

---

## Study Rationale

Botulinum toxin–induced muscle atrophy represents a controlled experimental model to study structural and temporal tissue changes. While histology provides a quantitative reference standard, it is invasive and unsuitable for longitudinal monitoring. Ultrasound radiomics combined with machine learning may offer a noninvasive alternative capable of detecting and temporally characterizing muscle atrophy.

This study evaluates whether radiomics-based models derived from ultrasound images (ERAT) can approximate or complement histology-based quantitative metrics (EHAT) in this context.

---

## Objectives

- To construct a histology-derived quantitative atrophy index (EHAT)
- To develop radiomics-based machine learning models from ultrasound images (ERAT)
- To evaluate ERAT performance for:
  - Classification of experimental groups
  - Regression of days post–BoNT-A injection
- To compare ERAT and EHAT using correlation, agreement, and group-wise statistical analyses
- To assess model interpretability using SHAP-based feature importance analysis

---

## Repository Structure

```text
├── notebooks/                 # Jupyter notebooks (analysis pipeline)
│   ├── 00_setup.ipynb
│   ├── 01_data_ingestion.ipynb
│   ├── 03_ehat_modeling.ipynb
│   ├── 04_erat_modeling.ipynb
│   └── 05_erat_vs_ehat_comparison.ipynb
├── src/                       # Reusable Python modules
│   ├── config.py
│   ├── io.py
│   ├── preprocessing.py
│   ├── stats.py
│   ├── metrics.py
│   ├── modeling.py
│   └── shap_utils.py
├── data/
│   ├── raw/                   # Raw data (not versioned)
│   └── processed/             # Processed datasets (generated locally)
├── results/
│   ├── figures/               # Main manuscript figures
│   ├── tables/                # Main manuscript tables
│   └── text/                  # Textual summaries of results
├── supplementary/
│   ├── shap/                  # SHAP interpretability outputs
│   ├── tables/
│   └── text/
├── requirements.txt           # Python dependencies
├── README.md
├── LICENSE
└── CITATION.cff


## Reproducibility

### Software Requirements
Python ≥ 3.13.11
All dependencies are specified in requirements.txt

### Setup
pip install -r requirements.txt

##EAnalysis Pipeline

All analyses are executed using Jupyter notebooks. The recommended execution order is:

00_setup.ipynb
Project configuration, environment checks, and reproducibility settings.

01_data_ingestion.ipynb
Loading, cleaning, and organization of raw ultrasound, histology, and metadata inputs.

03_ehat_modeling.ipynb
Construction and statistical analysis of the histology-based quantitative atrophy index (EHAT), including group-wise distribution analyses.

04_erat_modeling.ipynb
Development and evaluation of radiomics-based machine learning models (ERAT), including:

Multiclass classification of experimental groups

Regression of days post–BoNT-A injection

05_erat_vs_ehat_comparison.ipynb
Direct comparison between ERAT and EHAT using correlation, agreement metrics, Bland–Altman analysis, and group-wise statistical testing.

SHAP interpretability (executed within notebooks/05_erat_vs_ehat_comparison.ipynb and 07_shap_erat_regression.ipynb)
Model interpretability is performed using SHAP (SHapley Additive exPlanations) for:

LightGBM multiclass classification (global and class-specific feature importance)

LightGBM regression (global feature importance)

Outputs are written to supplementary/shap/ and are intended for supplementary reporting.

Each notebook is designed to be executed top-to-bottom.

##Outputs and Versioning Policy

GitHub repository
Contains code, notebooks, and minimal essential outputs required to understand and reproduce the workflow structure.
Results generated during execution (tables, figures, SHAP outputs):
Created locally under results/ and supplementary/
Not versioned on GitHub
Included in the Zenodo release associated with this repository

## Data Availability

Raw data are not publicly available due to experimental and ethical constraints.
Processed datasets required to reproduce the analyses are generated during the pipeline execution.

For access to raw data, please contact the corresponding author.

##Model Interpretability
Model interpretability is assessed using SHAP (SHapley Additive exPlanations):
Multiclass classification: global and class-specific feature importance
Regression models: global feature importance for continuous prediction

##Code Availability

All code used in this study is available in this repository under an open-source license.

##Citation

If you use this code, please cite the associated article and this repository (see CITATION.cff).