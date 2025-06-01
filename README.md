# Histopathologic-Cancer-Detection

**Identify metastatic tissue in histopathologic scans of lymph node sections**

---

## Problem Description

We participated in the Kaggle “Histopathologic Cancer Detection” competition, a binary classification task. Each sample in `train/` is a 96×96 RGB `.tif` patch extracted from whole‐slide histopathology scans. Our goal is to predict whether each patch contains metastatic tissue (`label = 1`) or not (`label = 0`).

---

## Files & Folders

- **`train_labels.csv`** (≈220k rows)  
  Contains two columns:  
  - `id` (40-character patch identifier)  
  - `label` (0 or 1)  

- **`train/`** (≈220k TIFF files)  
  Each named `<id>.tif`. These patches are labeled in `train_labels.csv`.  

- **`test/`** (≈57k TIFF files)  
  Each named `<id>.tif`. These patches are unlabeled and used for generating a Kaggle submission.  

- **`sample_submission.csv`**  
  Blank template for test IDs and placeholder labels.

---

## README Overview

Below is a summary of what we ultimately included in our Kaggle notebook and repository:

1. **Load and Inspect Labels**  
   - Read `train_labels.csv`  
   - Print total number of rows and count of each class (0 vs. 1)  

2. **Visualize Class Imbalance & Random Samples**  
   - Bar chart showing class distribution (≈130k non-metastasis vs. ≈89k metastasis)  
   - Display a handful of random 96×96 patches (positive/negative) from the `train/` folder  

3. **Compute Pixel-Level Statistics**  
   - Sample ~2,000 random patches to compute approximate per-channel mean & standard deviation  
   - Use these statistics to normalize images in both training and validation  

4. **Train/Validation Split**  
   - Perform an 80/20 stratified split on `train_labels.csv`  
   - Ensure both train/val sets preserve the same positive/negative proportion  

5. **Define PyTorch Dataset & DataLoader**  
   - Implement `HistopathFolderDataset` that reads patches from disk (no extra unzipping required)  
   - Apply data augmentations (random flips, normalization) to training set  
   - Use only resize + normalize for validation set  
   - Show number of training/validation batches per epoch  

6. **Quick EDA / Patch Visualization (Validation of Loading)**  
   - Display six random patches with their labels, skipping any missing/corrupted files  

7. **Model Architecture: TinyCNN**  
   - A lightweight CNN with two convolutional layers → max-pooling → small fully-connected head  
   - Input shape: (3×96×96) → output raw logit (single neuron)  

8. **Loss, Optimizer, and AUC Helper**  
   - Loss = `BCEWithLogitsLoss` (binary cross-entropy with logits)  
   - Optimizer = `Adam(lr=1e-4)`  
   - Helper function `compute_subset_auc(...)` to evaluate AUC on a small random subset (~512–1000 samples) for quick feedback  

9. **Full 5-Epoch Training Loop**  
   - Iterate over training batches, print batch loss every 500 steps  
   - After each epoch:  
     - Compute average training loss over the entire epoch  
     - Compute AUC on a small subset of training patches (≈1000)  
     - Compute AUC on a small subset of validation patches (≈1000)  
     - Print “Train Loss”, “TrainSubset AUC”, “ValSubset AUC” each epoch  
   - Record the best validation‐subset AUC (~0.9402 over 5 epochs)  

10. **Plot Training Curves**  
    - Training Loss vs. Epoch  
    - Subset AUC (Train vs. Val) vs. Epoch  

11. **Full Validation Evaluation**  
    - Run inference on entire validation set (all ≈44k images)  
    - Compute confusion matrix and classification report (precision/recall/F1 for each class)  
    - Report accuracy, precision, recall, F1, support  

12. **Test-Time Inference & Submission**  
    - Load all TIFF files from `test/`  
    - Apply the same validation transforms (resize + normalize)  
    - Run the trained model on each test sample to obtain a probability (`sigmoid(logit)`)  
    - Create `submission.csv` in the required format (`id,label`), sorted by test ID  

13. **Discussion / Conclusion**  
    - Summarize key findings:  
      - Class imbalance: ~60% negatives vs. 40% positives  
      - Simple CNN achieved ≈0.94 subset AUC on validation after 5 epochs  
      - Validation confusion matrix and classification report  
      - Next steps: deeper architectures (ResNet, EfficientNet), additional augmentations, learning-rate tuning, k-fold cross-validation, mixed precision training, fine-tuning a pretrained model  

---

## How to Use This Repository

1. **Open the Jupyter Notebook**  
   - You can run it end-to-end on Kaggle’s free GPU environment (or on your local machine if you have sufficient memory and a GPU).  
   - All code cells are organized in the order above.  

2. **Copy to Your Own GitHub**  
   - Download the `.ipynb` from Kaggle (via “File → Download” in the notebook menu).  
   - Create a new repository on GitHub (e.g., `histopathologic-cancer-detection`).  
   - Commit both the notebook (`.ipynb`) and this `README.md` with the exact folder structure:  
     ```
     /histopathologic-cancer-detection
       ├── Histopathologic_Cancer_Detection.ipynb
       ├── README.md
       ├── requirements.txt       # (if you wish to list dependencies)
       └── sample_submission.csv  # (optional reference)
     ```  
   - Push all files to GitHub.  

3. **Reproduce Results**  
   - On Kaggle: attach the “Histopathologic Cancer Detection” dataset in the Notebook’s “Data” panel.  
   - Run each cell in sequence; the notebook will automatically detect `/kaggle/input/histopathologic-cancer-detection/train/`, `/test/`, and `train_labels.csv`.  
   - At the end, you will have `submission.csv` ready to upload to the Kaggle competition.  


---

*End of README*
