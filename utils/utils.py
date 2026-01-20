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
    

# json내 dict접근을 namespace로 변환
def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    else:
        return d    
    

# load config
def load_cfg(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg_dict = json.load(f)
    return dict_to_namespace(cfg_dict)


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")
        
    return device
    

# split dataset
def split_dataset(dataset, labels, pids, random_state=42):
    # 0.6/0.2/0,2 split
    labels = np.array(labels)
    
    unique_pids = np.unique(pids)
    pid_labels = []
    

    for pid in unique_pids:
        pid_labels.append(labels[pids == pid][0])

    pid_labels = np.array(pid_labels)

    # 2. PID 단위 stratified split
    train_pids, temp_pids = train_test_split(
        unique_pids,
        test_size=0.4,
        random_state=random_state,
        stratify=pid_labels
    )

    # temp에 해당하는 pid_labels 다시 구성
    temp_labels = pid_labels[np.isin(unique_pids, temp_pids)]

    val_pids, test_pids = train_test_split(
        temp_pids,
        test_size=0.5,
        random_state=random_state,
        stratify=temp_labels
    )

    # 3. PID → sample index 매핑
    train_idx = np.where(np.isin(pids, train_pids))[0]
    val_idx   = np.where(np.isin(pids, val_pids))[0]
    test_idx  = np.where(np.isin(pids, test_pids))[0]

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx)
    )    
    
    
    # indices = np.arange(len(dataset))

    # train_idx, temp_idx = train_test_split(
    #     indices,
    #     test_size=0.4,
    #     random_state=random_state,
    #     stratify=labels
    # )

    # val_idx, test_idx = train_test_split(
    #     temp_idx,
    #     test_size=0.5,
    #     random_state=random_state,
    #     stratify=labels[temp_idx]
    # )

    # return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)