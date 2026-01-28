import os
import random
import numpy as np
import json
from types import SimpleNamespace

import torch
# from torch.utils.data import Subset
# from sklearn.model_selection import train_test_split



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
    
def namespace_to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {
            k: namespace_to_dict(v)
            for k, v in vars(obj).items()
        }
    elif isinstance(obj, dict):
        return {
            k: namespace_to_dict(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [namespace_to_dict(x) for x in obj]
    else:
        return obj
    
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


# EarlSyStopping
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def step(self, val_loss: float, patience: int) -> bool:
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
            # Exp Settings
            "learning_rate": exp_settings.learning_rate,
            "batch_size": exp_settings.batch_size,
            "num_epochs": exp_settings.num_epochs,
            "weight_decay": exp_settings.weight_decay,
            "num_workers": exp_settings.num_workers,
            "ealry_stopping_patience": exp_settings.ealry_stopping_patience,
            "min_delta": exp_settings.min_delta,
            
            # Model Settings
            "output_dim": model_settings.output_dim,
            "dropout": model_settings.dropout,
            "freeze_enc": args.freeze_enc,
            
            # File path
            "pretrained_model_path": pretrained_model_path,
            "train_csv_file": datasets_settings.train_csv_file,
            "valid_csv_file": datasets_settings.valid_csv_file,
            "test_csv_file": datasets_settings.test_csv_file,

            # Transform info            
            "image_size": transform_settings.image_size,
            
            # Model Saving Func
            "model_save_mode": model_save_mode,
            
            # ETC
            "etc": args.etc
        },
    )
    
    return wandb_run


def manage_args(mode='train'):
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="A simple command-line tool.")
    parser.add_argument("--cfg_path", default="./config.json", type=str, help="Data and Model hyperparameter Config JSON File Path.")
    parser.add_argument("--task", choices=["downstream_task1", "downstream_task2", "downstream_task3"], default="downstream_task1", help="Task to perform: upstream or downstream.")
    parser.add_argument("--pretrained_model", default="only_state_dict.pth.tar", help="pretrained_model.")
    
    parser.add_argument("--argument_dir", default=None, help="Using Argument Data (Using: --argument_data 'paired_data')")

    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")

    # Overwrite to use colab
    parser.add_argument("--data_root_path", default=None, type=str, help="Override data root path in config.")
    
    
    # Overwirte to Exp hyperparameter
    parser.add_argument("--learning_rate", default=None, type=float, help="Override learning rate in config.")
    parser.add_argument("--weight_decay", default=None, type=float, help="Override weight decay in config.")
    parser.add_argument("--batch_size", default=None, type=int, help="Override batch size in config.")
    parser.add_argument("--num_epochs", default=None, type=int, help="Override number of epochs in config.")
    parser.add_argument("--early_stopping_patience", default=None, type=int, help="Override early stopping patience in config.")
    
    # Overwirte to Model hyperparameter
    parser.add_argument("--freeze_enc", action="store_true", help="Freezing Encoder Head(If turn on: Train Enc)")
    
    
    # 실험이 안정적이여서 모델 저장할 때 사용
    parser.add_argument("--save_model", action="store_true", help="Saving the model(Using: --save_model).")
    parser.add_argument("--wandb", action="store_true", help="Using wandb logging(Using: --wandb).")    
    
    
    # 기타로 뭔가 내용 적고 싶을 때
    parser.add_argument("--etc", default=None, type=str, help="Note Something")
    
    
    if mode=='test':
        # Test
        parser.add_argument("--best_model", default=None, required=True, type=str, help="best model checkpoint name for testing. (Using: model.pth)")
        parser.add_argument("--only_inference", action="store_true", help="Only inference Not save test result")
        
        # Reverse X1, X2 path
        # 바뀐 인풋에 대해서 아웃풋이 반대로 나오는지 확인하기 위함
        parser.add_argument("--reverse", action="store_true", help='Change X1, X2 Path for check time effect')
        
    
    
    return parser