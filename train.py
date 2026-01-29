import torch
from tqdm import tqdm
import os
from utils.utils import *
from utils.measure import evaluate_model


def train(args, cfg, wandb_run, dataloader, model, optimizer, criterion, device="cpu"):
    now = get_time()
    
    # save model
    save_model_root_path = getattr(cfg, "save_model_root_path", None)
    task = args.task
    os.makedirs(os.path.join(save_model_root_path, task), exist_ok=True)
    
    save_path = os.path.join(save_model_root_path, task, now + ".pth")
    print()
    print('--'*20)
    print(f"Save Path: {save_path}")
    print('--'*20)
    print()
    
    # early stopping settings
    patience = getattr(cfg.exp_settings, "ealry_stopping_patience", 10)
    min_delta = getattr(cfg.exp_settings, "early_stopping_min_delta", 1e-5)
    early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)
    
    # exp settings
    num_epochs = cfg.exp_settings.num_epochs
    best_val_loss = float('inf')
    best_roc_auc = -1
    best_epoch = -1


    # train & valid loop
    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for data in tqdm(dataloader['train'], desc=f"Epoch {epoch+1}/{num_epochs} Training"):
            x1, x2, target = data["x1"].to(device), data["x2"].to(device), data["y"].to(device)
            
            optimizer.zero_grad()
            out = model(x1, x2)
                        
            if cfg.model.output_dim == 1:
                logits = out.squeeze(1)
                loss = criterion(logits, target.float())
            else:
                logits = out
                loss = criterion(logits, target.long())
            
            loss.backward()
            optimizer.step()
            
            bs = target.size(0)
            train_loss_sum += loss.item() * bs
            train_count += bs

        train_loss = train_loss_sum / max(train_count, 1)
        print()
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.3f}")
        
        # validation
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        all_outputs = []
        all_targets = []
        
        best_youden = -float('inf')
        best_info = {}
        
        with torch.no_grad():
            for data in tqdm(dataloader['val'], desc=f"Epoch {epoch+1}/{num_epochs} Validation"):
                x1, x2, target = data["x1"].to(device), data["x2"].to(device), data["y"].to(device)

                out = model(x1, x2)
                                
                if cfg.model.output_dim == 1:
                    logits = out.squeeze(1)
                    val_loss = criterion(logits, target.float())
                    probs = torch.sigmoid(logits)
                else:
                    logits = out
                    val_loss = criterion(logits, target.long())
                    probs = torch.softmax(logits, dim=1)[:, 1]
                                    
                
                bs = target.size(0)
                val_loss_sum += val_loss.item() * bs
                val_count += bs
                
                all_outputs.append(probs)
                all_targets.append(target)
                
        val_loss = val_loss_sum / max(val_count, 1)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        roc_auc, accuracy, f1, sensitivity, specificity, best_threshold = evaluate_model(all_outputs, all_targets)

        print(f"Validation Loss: {val_loss:.3f}")
        print(f"ROC AUC: {roc_auc:.3f}, Accuracy: {accuracy:.3f}, F1: {f1:.3f}, "
              f"Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}, Best_TH: {best_threshold:.2f}")
        
        # youden: TPR - FPR이 최대가 되는 점 (즉, 하나의 th에서 sensitivity와 specificity의 균형이 얼마나 좋은지 보는 지표)
        youden = sensitivity + specificity - 1
        
        # Save best
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_youden = max(best_youden, youden)
            best_epoch = epoch + 1
            best_info = {
                "epoch": best_epoch,
                "val_loss": float(val_loss),
                "roc_auc": float(roc_auc),
                "accuracy": float(accuracy),
                "f1": float(f1),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "youden": float(youden),
                "best_threshold": float(best_threshold),
            }            
            
            if getattr(args, "save_model", False):
                print()
                ckpt = {
                    "state_dict": model.state_dict(),
                    "epoch": best_epoch,
                    "metrics": best_info,
                    "args": vars(args),
                    "cfg" : namespace_to_dict(cfg)
                    
                }
                torch.save(ckpt, save_path)
                print("Best model saved.")
        
        if early_stopper.step(val_loss, patience):
            print()
            print(f"Early stop at {epoch+1}")
            break
        
        if wandb_run:
            wandb_run.log({
                "Train Loss": train_loss,
                "Validation Loss": val_loss,
                "ROC AUC": roc_auc,
                "Accuracy": accuracy,
                "F1 Score": f1,
                "Sensitivity": sensitivity,
                "Specificity": specificity,
                "Youden": youden,
                "Best_Threshhold": best_threshold
            })
        
        print()

    print('--'* 20)
    print("Training completed.")
    print()
    print(f"Save Path: {save_path}")
    print('--'* 20)
    
    if wandb_run:
        wandb_run.finish()