import os
import torch
from tqdm import tqdm
import pandas as pd

from utils.utils import *
from utils.measure import evaluate_model
from utils.transform import *

from dataset import PFRatioDataset
from torch.utils.data import DataLoader

from monai.data import pad_list_data_collate
from models.model import DownStreamTaskModel

def stack_inference(results, pid, conf, preds, y, path1, path2):
    if torch.is_tensor(y):
        y_list = y.detach().cpu().view(-1).tolist()
    else:
        y_list = list(y)

    # pid/idx도 tensor일 수 있으니 정리
    if torch.is_tensor(pid):
        pid_list = pid.detach().cpu().view(-1).tolist()
    else:
        pid_list = list(pid)

    for i in range(len(preds)):
        results.append({
            "PID": int(pid_list[i]),
            "pred": float(preds[i].item()),
            "label": int(y_list[i]),
            "confidence": float(conf[i].item()),
            "cxr path1": path1[i],
            "cxr path2": path2[i],
        })
        
    return results


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
    parser = manage_args(mode='test')
    args = parser.parse_args()
    
    
    args = parser.parse_args()
    fix_seed(args.seed)
    
    device = get_device()

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
        cfg.dataset.test_csv_file = os.path.join(args.argument_dir, cfg.dataset.test_csv_file)

    test_ds = PFRatioDataset(cfg=cfg, reverse=args.reverse, flag="test", transform=transform)
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
    ckpt_path = os.path.join('./save', args.task, best_model)
    
    print("Check Point Path: ", ckpt_path)

    model = DownStreamTaskModel(args, cfg)
    model = load_checkpoint(model, ckpt_path, device=device, strict=True)
    print(f"Loaded checkpoint: {ckpt_path}")

    model.to(device)
    model.eval()

    all_outputs = []
    all_targets = []

    results = []
    with torch.no_grad():
        for data in tqdm(dataloader["test"], desc="Testing"):
            x1 = data["x1"].to(device)
            x2 = data["x2"].to(device)
            target  = data["y"]            # cpu여도 됨
            pid = data["pid"]
            path1 = data["cxr_path1"]
            path2 = data["cxr_path2"]
            
            out = model(x1, x2)
            logits = out.squeeze(1)

            probs = torch.sigmoid(logits).detach().cpu().view(-1)
            preds = (probs >= 0.5).long() 
            
            conf = torch.where(preds == 1, probs, 1 - probs)
            
            all_outputs.append(probs)
            all_targets.append(target.detach().cpu().view(-1))
            
            results = stack_inference(results, pid, conf, preds, target, path1, path2)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    roc_auc, accuracy, f1, sensitivity, specificity, best_threshold = evaluate_model(all_outputs, all_targets)
    print(f"ROC AUC: {roc_auc:.4f}, Accuracy: {accuracy:.4f}, Sensitivity: {sensitivity:.4f}, "
            f"Specificity: {specificity:.4f}, F1: {f1:.4f}, Best_TH: {best_threshold:.2f}")

    # csv 저장
    # cols = ['best_model','roc_auc','accuracy','f1','sensitivity','specificity']
    os.makedirs("./res", exist_ok=True)
    
    if args.task == "downstream_task1":
        file_nm = "res_task1"
    elif args.task == "downstream_task2":
        file_nm = "res_task2"
    elif args.task == "downstream_task3":
        file_nm = "res_task3"
    save_test_res_path = os.path.join("./res", args.task, f"{file_nm}.csv")
    
    # inference를
    if not getattr(args, "only_inference", False):
        # csv가 없으면 헤더부터 저장
        if not os.path.exists(save_test_res_path):
            pd.DataFrame(columns=['best_model', 'roc_auc','accuracy','sensitivity','specificity','f1', 'best_thershhold', 'etc']).to_csv(save_test_res_path, index=False)
            
        # 이후 결과 추가 저장
        pd.DataFrame(
            [[args.best_model, f"{roc_auc:.4f}", f"{accuracy:.4f}", f"{sensitivity:.4f}", f"{specificity:.4f}", f"{f1:.4f}", f"Best_TH: {best_threshold:.2f}", f"{args.etc}"]],
            columns=['best_model','roc_auc','accuracy','sensitivity','specificity','f1', 'best_thershhold', 'etc']
        ).to_csv(save_test_res_path, mode='a', header=False, index=False)
        
    
    if getattr(args, "reverse", False):
        inference_nm = os.path.join("./res", args.task, "R_" + args.best_model.replace(".pth", ".csv"))
    else:
        inference_nm = os.path.join("./res", args.task, args.best_model.replace(".pth", ".csv"))
    pd.DataFrame(results).to_csv(inference_nm, index=False)    
        
                
if __name__=='__main__':
    test()