from argparse import ArgumentParser
from dataset import PFRatioDataset
from train import train
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
    parser.add_argument("--task", choices=["downstream_task1", "downstream_task2", "downstream_task3"], default="downstream_task1", help="Task to perform: upstream or downstream.")
    parser.add_argument("--pretrained_model", default="only_state_dict.pth.tar", help="pretrained_model.")

    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")

    # Overwrite to use colab
    parser.add_argument("--data_root_path", default=None, type=str, help="Override data root path in config.")
    
    
    # Overwirte to model hyperparameter
    parser.add_argument("--learning_rate", default=None, type=float, help="Override learning rate in config.")
    parser.add_argument("--weight_decay", default=None, type=float, help="Override weight decay in config.")
    parser.add_argument("--batch_size", default=None, type=int, help="Override batch size in config.")
    parser.add_argument("--num_epochs", default=None, type=int, help="Override number of epochs in config.")
    parser.add_argument("--early_stopping_patience", default=None, type=int, help="Override early stopping patience in config.")
    
    
    # 실험이 안정적이여서 모델 저장할 때 사용
    parser.add_argument("--save_model", action="store_true", help="Saving the model(Using: --save_model).")
    parser.add_argument("--wandb", action="store_true", help="Using wandb logging(Using: --wandb).")
    
    args = parser.parse_args()
    fix_seed(args.seed)

    cfg = load_cfg(args.cfg_path)
    if args.task == "downstream_task1":
        cfg = cfg.downstream_task1
    elif args.task == "downstream_task2":
        cfg = cfg.downstream_task2
    elif args.task == "downstream_task3":
        cfg = cfg.downstream_task3
        
    cfg.pretrained_model_path = os.path.join("models", args.pretrained_model)
        
        
    transform = get_transform(cfg)
    
    cfg.data_root_path = args.data_root_path if args.data_root_path is not None else cfg.data_root_path
    cfg.exp_settings.learning_rate = args.learning_rate if args.learning_rate is not None else cfg.exp_settings.learning_rate
    cfg.exp_settings.weight_decay = args.weight_decay if args.weight_decay is not None else cfg.exp_settings.weight_decay
    cfg.exp_settings.batch_size = args.batch_size if args.batch_size is not None else cfg.exp_settings.batch_size
    cfg.exp_settings.num_epochs = args.num_epochs if args.num_epochs is not None else cfg.exp_settings.num_epochs
    cfg.exp_settings.early_stopping_patience = args.early_stopping_patience if args.early_stopping_patience is not None else cfg.exp_settings.ealry_stopping_patience
        
    print("Configuration Loaded:", cfg) 
        
    wandb_run = None
    if getattr(args, "wandb", False):
        wandb_run = wandb_init(args, cfg)
    
    # Data Settings
    train_ds = PFRatioDataset(cfg=cfg, flag="train",transform=transform)
    val_ds = PFRatioDataset(cfg=cfg, flag="valid",transform=transform)
    
    # labels = full_dataset.df["SIMPLE LABEL"].to_numpy()
    # pids = full_dataset.df["PID"].values
    # train_ds, val_ds, test_ds = split_dataset(full_dataset, labels, pids, random_state=42)
    
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
    
    dataloader = {
        "train": train_loader,
        "val": val_loader,
    }
    
    # Model Settings
    model = DownStreamTaskModel(cfg)
    device = get_device()
    model.to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg.exp_settings.learning_rate,
        weight_decay=cfg.exp_settings.weight_decay
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    # Train
    train(args, cfg, wandb_run, dataloader, model, optimizer, criterion, device)
    
if __name__ == "__main__":
    main()