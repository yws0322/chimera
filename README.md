# MICCAI 2025 CHIMERA Challenge Task 1 2nd Place Solution

## Challenge Overview

This repository contains our 2nd place solution for the MICCAI 2025 CHIMERA Challenge Task 1, which focuses on predicting time to biochemical recurrence (BCR) or follow-up in prostate cancer patients after prostatectomy using the Concordance Index (C-index) as the evaluation metric.

## Solution Overview

This directory contains a complete pipeline for training and evaluating survival models on clinical data. The pipeline consists of three main steps:
1. **Fold Creation**: Create stratified K-fold splits and save as CSV
2. **Model Training**: Train 6 models per fold and save top-3 performers
3. **Ensemble Evaluation**: Load all saved models and evaluate ensemble performance

## File Structure

```
chimera/
├── create_fold.py          # Step 1: Create folds with stratification
├── unimodel_main.py        # Step 2: Train models and save top-3 per fold
├── ensemble_main.py        # Step 3: Evaluate ensemble of all saved models
├── models/                 # Model implementations
│   ├── mlp_model.py
│   ├── real_mlp_model.py
│   ├── tabm_model.py
│   └── traditional_model.py
└── README.md 
```

### Required Python Packages

**Core dependencies:**
```bash
pip install numpy pandas scikit-learn scikit-survival torch xgbse tabm rtdl-num-embeddings
```

## Usage

### Step 1: Create Folds

Creates CSV files with fold assignments using uniform stratification.

```bash
cd chimera
python create_fold.py
```

**Output:**
- `clinical_data_with_folds_age.csv` - Stratified by age at prostatectomy
- `clinical_data_with_folds_followup.csv` - Stratified by follow-up time

### Step 2: Train Models

Trains 6 models on each fold and saves the top-3 performers.

```bash
python unimodel_main.py
```

**Models Trained:**
1. MLP
2. RealMLP
3. TabM 
4. CoxPH 
5. RandomSurvivalForest 
6. SurvivalXGBoost 

**Output:**
- `saved_models/followup_fold_XX/` - Top-3 models per fold
- `results/unimodel_evaluation_results.json` - Training results summary

**Total Models Saved:**
- 10 folds × 3 models = 30 models (for single split)
- With age split: 10 folds × 3 models × 2 splits = 60 models

### Step 3: Evaluate Ensemble

Loads all saved models and evaluates ensemble performance.

```bash
python ensemble_main.py
```

**Ensemble Method:**
- Simple Average: Equal-weighted average of all model predictions

**Output:**
- `results/ensemble_evaluation_results.json` - Ensemble evaluation results

## Results

| Model | CV Test C-index | Public Leaderboard |
|-------|-----------------|-------------------|
| RandomSurvivalForest | 0.965 ± 0.042 | - |
| TabM | 0.939 ± 0.055 | 0.7273 |
| RealMLP | 0.879 ± 0.122 | - |
| **MLP** | 0.820 ± 0.123 | **0.7521** (2nd) |
| CoxPH | 0.761 ± 0.279 | - |
| XGBoost | 0.616 ± 0.359 | - |
| Ensemble (60 models) | 0.963 ± 0.042 | 0.7438 |

**Note**: CV = Cross-Validation (10-fold on followup split)

## Citations

If you use this repository in your research, please cite:

```bibtex
@software{chimera2025,
  author = {Your Name},
  title = {CHIMERA: Clinical Survival Model Pipeline},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yws0322/chimera}
}
```

This project uses implementations from the following works:
```bibtex
@software{realmlp_standalone,
  author = {Holzmueller, David},
  title = {RealMLP-TD-S Standalone Implementation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/dholzmueller/realmlp-td-s_standalone}
}
```
```bibtex
@inproceedings{tabm2025,
  title={TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling},
  author={Gorishniy, Yury and Rubachev, Ivan and Kartashev, Nikolay and Shlenskii, Daniil and Kodryan, Akim and Babenko, Artem},
  booktitle={International Conference on Learning Representations},
  year={2025},
  url={https://arxiv.org/abs/2410.24210}
}
```

## License

This project builds upon open-source implementations:
- RealMLP-TD-S: MIT License
- TabM: Apache-2.0 License

Please cite the original works if you use these models in your research.
