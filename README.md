# Minimal Learning Machine for Multi-Label Learning

MATLAB implementations of Minimal Learning Machine (MLM) approaches and metrics for multi-label classification. 

## Sample script for training and testing
To run training and testing for the approaches see `run_ml_mlm_demo.m`. This script uses a synthetic dataset. 

## Training
 - Distance regression training (MLM training) `dist_reg_train.m`
 - LOOCV with Ranking Loss statistic for ML-MLM `ml_mlm_loocv_train.m`
 
## Predict
 - Nearest Neigbour MLM (NN-MLM) `nn_mlm_pred.m`
 - Localization Linear System MLM (LLS-MLM) `lls_mlm_pred.m` 
 - Cubic equations MLM (C-MLM) `cubic_mlm_pred.m`
 - Multi-Label MLM `ml_mlm_pred.m`

## Thresholding
- Local Rcut thresholding `local_rcut.m`

## Metrics
- Compute all metric results: `compute_metrics.m` (see `run_ml_mlm_demo.m` for examples)
- Ranking: `ranking_loss.m`, `average_precision.m`, `coverage.m`, `one_error.m`, `precision_at_k.m`
- Bipartition: `accuracy.m`, `hamming_loss.m`, `macro_f1`, `micro_f1`, `micro_recall`, `micro_precision`