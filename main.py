from argparse import ArgumentParser
from dataset import PFRatioDataset
from train import train
# from test import test
from models.model import DownStreamTaskModel
from utils.utils import *
from utils.transform import *

from torch.utils.data import DataLoader
# 크기가 제각각이면 pad_list_data_collate가 더 안전
from monai.data import pad_list_data_collate

from warnings import filterwarnings
filterwarnings("ignore")

def main():
    parser = ArgumentParser(description="A simple command-line tool.")
    parser.add_argument("--cfg_path", default="./config.json", type=str, help="Data and Model hyperparameter Config JSON File Path.")
    parser.add_argument("--task", choices=["upstream", "downstream"], default="downstream", help="Task to perform: upstream or downstream.")
    parser.add_argument("--data_root_path", default=None, type=str, help="Override data root path in config.")

    parser.add_argument("--is_save_model", type=bool, help="Whether to save the trained model.")
    
    # early stopping
    parser.add_argument("--early_stopping_patience", default=5, type=int, help="Early stopping patience.")
    
    
    args = parser.parse_args()
    fix_seed(42)

    cfg = load_cfg(args.cfg_path)
    if args.task == "downstream":
        cfg = cfg.downstream
    else:
        cfg = cfg.upstream
    
    transform = get_transform(cfg)
    
    if args.data_root_path is not None:
        cfg.data_root_path = args.data_root_path
    
    # Data Settings
    full_dataset = PFRatioDataset(cfg=cfg, transform=transform)
    
    labels = full_dataset.df["SIMPLE LABEL"].to_numpy()
    pids = full_dataset.df["PID"].values
    train_ds, val_ds, test_ds = split_dataset(full_dataset, labels, pids, random_state=42)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.exp_settings.batch_size,
        shuffle=True,
        num_workers=cfg.exp_settings.num_workers,
        collate_fn=pad_list_data_collate,           # 크기 다른 샘플 섞일 수 있으면 이걸로
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.exp_settings.batch_size,
        shuffle=False,
        num_workers=cfg.exp_settings.num_workers,
        collate_fn=pad_list_data_collate,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.exp_settings.batch_size,
        shuffle=False,
        num_workers=cfg.exp_settings.num_workers,
        collate_fn=pad_list_data_collate,
    )    
    
    dataloader = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
    
    # Model Settings
    model = DownStreamTaskModel(cfg.model)
    device = get_device()
    model.to(device)
    
    # Load pretrained weights
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.exp_settings.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Train
    train(args, cfg, dataloader, model, optimizer, criterion, device)
    
    # Evaluate

if __name__ == "__main__":
    main()