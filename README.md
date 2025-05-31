# Histopathologic-Cancer-Detection
Identify metastatic tissue in histopathologic scans of lymph node sections

**Problem Description**  
We are participating in the Kaggle “Histopathologic Cancer Detection” competition, a binary classification task.  
Each sample in `train/` is a 96×96 RGB `.tif` patch extracted from whole‐slide histopathology scans.  
Our goal is to predict whether each patch contains metastatic tissue (`label=1`) or not (`label=0`).  

**Files & Folders**  
- `train_labels.csv` (≈220k rows): contains two columns: `id` and `label` (0 or 1).  
- `train/` (≈220k TIFF files): each named `<id>.tif`.  
- `test/` (≈57k TIFF files): each named `<id>.tif`, to be scored and submitted for Kaggle.  
- `sample_submission.csv`: blank template for test IDs and placeholder labels.  

Below we will:
1. Load and inspect `train_labels.csv`.  
2. Visualize class imbalance and sample images.  
3. Perform pixel‐level EDA (histograms, summary statistics).  
4. Split data into train/validation.  
5. Build simple CNN and a ResNet50‐based model.  
6. Train, evaluate, and compare.  
7. Generate final `submission.csv`.
