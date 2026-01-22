import os
import random
import numpy as np
import pandas as pd
import json
from types import SimpleNamespace

import torch
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
load_dotenv("wandb.env")

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
    
    
def get_time():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%m%d_%H%M")
    
    return now

# # split dataset
# """
# PID를 기준으로 나누되, 각 PID의 slice당 레이블의 분포 차이를 stratified하게 유지하도록 나눔
# (Deprecated)
# """
# def split_dataset(dataset, labels, pids, random_state=42):
#     # 0.8/0.1/0,1 split
#     labels = np.array(labels)
    
#     unique_pids = np.unique(pids)
#     pid_labels = []
    
#     for pid in unique_pids:
#         pid_labels.append(labels[pids == pid][0])

#     pid_labels = np.array(pid_labels)

#     # PID 단위 stratified split
#     train_pids, temp_pids = train_test_split(
#         unique_pids,
#         test_size=0.2,
#         random_state=random_state,
#         stratify=pid_labels
#     )

#     # temp에 해당하는 pid_labels 다시 구성
#     temp_labels = pid_labels[np.isin(unique_pids, temp_pids)]

#     val_pids, test_pids = train_test_split(
#         temp_pids,
#         test_size=0.5,
#         random_state=random_state,
#         stratify=temp_labels
#     )

#     # 3. PID → sample index 매핑
#     train_idx = np.where(np.isin(pids, train_pids))[0]
#     val_idx   = np.where(np.isin(pids, val_pids))[0]
#     test_idx  = np.where(np.isin(pids, test_pids))[0]

#     return (
#         Subset(dataset, train_idx),
#         Subset(dataset, val_idx),
#         Subset(dataset, test_idx)
#     )    
    
    
# EarlSyStopping
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def step(self, val_loss: float, patience) -> bool:
        # if return True stop
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            print(f"Early stopping triggered {self.counter}/{patience}")
            return self.counter >= self.patience
        

def wandb_init(args, cfg):
    import wandb
    os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
    
    name = f"{get_time()}_{args.task}"
    
    pretrained_model_path = cfg.pretrained_model_path
    exp_settings = cfg.exp_settings
    model_settings = cfg.model
    transform_settings = cfg.transform
    datasets_settings = cfg.dataset
    model_save_mode = args.save_model
    
    wandb_run = wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project=os.getenv("WANDB_PROJECT"),
        name=name,
        
        config={
            "learning_rate": exp_settings.learning_rate,
            "batch_size": exp_settings.batch_size,
            "num_epochs": exp_settings.num_epochs,
            "weight_decay": exp_settings.weight_decay,
            "num_workers": exp_settings.num_workers,
            "ealry_stopping_patience": exp_settings.ealry_stopping_patience,
            "min_delta": exp_settings.min_delta,
            
            "pretrained_model_path": pretrained_model_path,
            
            "train_csv_file": datasets_settings.train_csv_file,
            "valid_csv_file": datasets_settings.valid_csv_file,
            "test_csv_file": datasets_settings.test_csv_file,
            
            "image_size": transform_settings.image_size,
            "model_save_mode": model_save_mode
        },
    )
    
    return wandb_run