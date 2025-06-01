# ─── Cell 1: Imports ────────────────────────────────────────────────────────────
import os
import random
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# PyTorch imports
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Scikit‐learn (for metrics, train/test split)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

%matplotlib inline
