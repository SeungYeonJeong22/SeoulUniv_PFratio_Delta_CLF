import os
import torch
from tqdm import tqdm

from utils.utils import *
from utils.measure import evaluate_model
from utils.transform import *

from argparse import ArgumentParser
from dataset import PFRatioDataset
from torch.utils.data import DataLoader

from monai.data import pad_list_data_collate
from models.model import DownStreamTaskModel


def load_checkpoint(model, ckpt_path: str, device="cpu", strict: bool = True):
    if ckpt_path is None:
        raise ValueError("Check model path is None.")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=strict)
    return model


def test():
    parser = ArgumentParser(description="A simple command-line tool.")
    parser.add_argument("--cfg_path", default="./config.json", type=str, help="Data and Model hyperparameter Config JSON File Path.")
    parser.add_argument("--task", choices=["upstream", "downstream_task1"], default="downstream_task1", help="Task to perform: upstream or downstream.")

    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")

    # Overwrite to use colab
    parser.add_argument("--data_root_path", default=None, type=str, help="Override data root path in config.")
    
    parser.add_argument("--best_model", default=None, required=True, type=str, help="best model checkpoint name for testing. (Using: model.pth)")
    
    args = parser.parse_args()
    
    
    args = parser.parse_args()
    fix_seed(args.seed)
    
    device = get_device()

    cfg = load_cfg(args.cfg_path)
    if args.task == "downstream_task1":
        cfg = cfg.downstream_task1
        
    transform = get_transform(cfg)
    
    if args.data_root_path is not None:
        cfg.data_root_path = args.data_root_path    

    test_ds = PFRatioDataset(cfg=cfg, flag="test",transform=transform)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.exp_settings.batch_size,
        shuffle=False,
        num_workers=cfg.exp_settings.num_workers,
        collate_fn=pad_list_data_collate,
    )    
    
    dataloader = {
        "test": test_loader
    }    
    
    best_model = args.best_model
    ckpt_path = os.path.join('./save', best_model)
    
    print(ckpt_path)

    model = DownStreamTaskModel(cfg)
    model = load_checkpoint(model, ckpt_path, device=device, strict=True)
    print(f"Loaded checkpoint: {ckpt_path}")

    model.to(device)
    model.eval()

    test_loss_sum = 0.0
    test_count = 0

    all_outputs = []
    all_targets = []
    all_pids = []  

    with torch.no_grad():
        for data in tqdm(dataloader["test"], desc="Testing"):
            x1 = data["x1"].to(device)
            x2 = data["x2"].to(device)
            target = data['y'].to(device)
            pid = data['pid']
            
            out = model(x1, x2)
            logits = out.squeeze(1)

            probs = torch.sigmoid(logits)
            all_outputs.append(probs.detach().cpu())
            all_targets.append(target.detach().cpu())
            all_pids.append(pid)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    roc_auc, accuracy, f1, sensitivity, specificity = evaluate_model(all_outputs, all_targets)
    print(f"ROC AUC: {roc_auc:.4f}, Accuracy: {accuracy:.4f}, Sensitivity: {sensitivity:.4f}, "
            f"Specificity: {specificity:.4f}, F1: {f1:.4f}")

    # csv 저장
    # cols = ['best_model','roc_auc','accuracy','f1','sensitivity','specificity']
    os.makedirs("./res", exist_ok=True)
    save_test_res_path = os.path.join("./res", "test_results.csv")
    
    # csv가 없으면 헤더부터 저장
    if not os.path.exists(save_test_res_path):
        pd.DataFrame(columns=['best_model','roc_auc','accuracy','sensitivity','specificity','f1']).to_csv(save_test_res_path, index=False)
        
    # 이후 결과 추가 저장
    pd.DataFrame(
        [[args.best_model, f"{roc_auc:.4f}", f"{accuracy:.4f}", f"{sensitivity:.4f}", f"{specificity:.4f}", f"{f1:.4f}"]],
        columns=['best_model','roc_auc','accuracy','sensitivity','specificity','f1']
    ).to_csv(save_test_res_path, mode='a', header=False, index=False)
    
    
    

    
    
    # return {
    #     "roc_auc": roc_auc,
    #     "accuracy": accuracy,
    #     "f1": f1,
    #     "sensitivity": sensitivity,
    #     "specificity": specificity,
    #     "probs": all_outputs,
    #     "targets": all_targets,
    #     "ids": all_pids if len(all_pids) > 0 else None
    # }

        
if __name__=='__main__':
    test()