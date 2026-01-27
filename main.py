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
    parser = manage_args()
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
    
    if args.argument_dir:
        cfg.dataset.train_csv_file = os.path.join(args.argument_dir, cfg.dataset.train_csv_file)
        cfg.dataset.valid_csv_file = os.path.join(args.argument_dir, cfg.dataset.valid_csv_file)
    
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
    model = DownStreamTaskModel(args, cfg)
    device = get_device()
    model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.exp_settings.learning_rate,
        weight_decay=cfg.exp_settings.weight_decay
    )
    
    if cfg.model.output_dim == 1: criterion = torch.nn.BCEWithLogitsLoss()
    else: criterion = torch.nn.CrossEntropyLoss()

    # Train
    train(args, cfg, wandb_run, dataloader, model, optimizer, criterion, device)
    
if __name__ == "__main__":
    main()