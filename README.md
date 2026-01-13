# Radiomics-Based Detection of Masseter Muscle Atrophy Using Ultrasound and Machine Learning

This repository contains the full data analysis pipeline for a study investigating the use of radiomics and machine learning applied to ultrasound images to detect and temporally characterize masseter muscle atrophy induced by botulinum toxin type A (BoNT-A), using histology as a comparative reference.

The project follows the ARRIVE, STARD, and CLAIM guidelines for experimental design, analysis, and reporting.

---

## Project Overview

**Objective**

To compare the performance of radiomics-based machine learning models derived from ultrasound images (ERAT) with histological quantitative measurements (EHAT) for identifying and timing masseter muscle atrophy after BoNT-A injection.

---

## Repository Structure
├── notebooks/ # Jupyter notebooks (analysis pipeline)
├── src/ # Reusable Python functions
├── data/
│ ├── raw/ # Raw data (not versioned)
│ └── processed/ # Processed datasets
├── results/
│ ├── figures/ # Main figures (manuscript)
│ ├── tables/ # Main tables (manuscript)
│ └── text/ # Text summaries of results
├── supplementary/
│ ├── shap/ # SHAP interpretability outputs
│ ├── tables/
│ └── text/
├── requirements.txt # Python dependencies
├── README.md
├── LICENSE
└── CITATION.cff

## Reproducibility

### Requirements
- Python >= 3.11
- See `requirements.txt` for all dependencies.

### Setup
```bash
pip install -r requirements.txt

##Execution order (Jupyter notebooks)

00_setup.ipynb
01_data_ingestion.ipynb
02_ehat_construction.ipynb
03_erat_models.ipynb
04_erat_ehat_comparison.ipynb
05_shap.ipynb

#Each notebook can be executed top-to-bottom to reproduce the corresponding results.

## Data Availability

Raw data are not publicly available due to experimental and ethical constraints.
Processed datasets required to reproduce the analyses are generated during the pipeline execution.

For access to raw data, please contact the corresponding author.

##Code Availability

All code used in this study is available in this repository under an open-source license.

##Citation

If you use this code, please cite the associated article and this repository (see CITATION.cff).