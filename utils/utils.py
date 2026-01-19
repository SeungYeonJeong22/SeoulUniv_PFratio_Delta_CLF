import os
import random
import numpy as np
import pandas as pd
import json
from types import SimpleNamespace

import torch
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

# seed fix
def fix_seed(seed: int = 42):
    # 1) Python
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # 2) NumPy
    np.random.seed(seed)

    # 3) PyTorch (CPU/GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

# load config
def load_cfg(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg_dict = json.load(f)
    return SimpleNamespace(**cfg_dict)    
    

# split dataset
def split_dataset(dataset, labels, random_state=42):
    # 0.6/0.2/0,2 split
    indices = np.arange(len(dataset))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.4,
        random_state=random_state,
        stratify=labels
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=random_state,
        stratify=labels[temp_idx]
    )

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)