from argparse import ArgumentParser
from dataset import PFRatioDataset
from train import train
# from test import test
from models.model import DownStreamTaskModel
from utils.utils import *
from utils.transform import *

from torch.utils.data import DataLoader
from warnings import filterwarnings
filterwarnings("ignore")

def main():
    parser = ArgumentParser(description="A simple command-line tool.")
    parser.add_argument("--cfg_path", default="./config.json", type=str, help="Data and Model hyperparameter Config JSON File Path.")
    
    # Experiment settings
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--num_epochs", default=10, type=int, help="Number of epochs to train.")
    
    # early stopping
    parser.add_argument("--early_stopping_patience", default=5, type=int, help="Early stopping patience.")
    
    
    args = parser.parse_args()
    fix_seed(42)
    
    cfg = load_cfg(args.cfg_path)
    
    transform = get_transform()
    
    # Data Settings
    full_dataset = PFRatioDataset(cfg=cfg, transform=transform)
    
    labels = full_dataset.patient_info["SIMPLE LABEL"].to_numpy()
    train_set, val_set, test_set = split_dataset(full_dataset, labels, random_state=42)
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    dataloader = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
    
    # Model Settings
    model = DownStreamTaskModel(enc_dims=512, output_dim=2)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Train
    train(args, cfg, dataloader, model, optimizer, criterion, device)
    
    # Evaluate

if __name__ == "__main__":
    main()