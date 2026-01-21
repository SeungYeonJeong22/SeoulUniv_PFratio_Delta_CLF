import torch
from tqdm import tqdm
from datetime import datetime
import os
from utils.utils import EarlyStopping
from utils.measure import evaluate_model

def train(args, cfg, dataloader, model, optimizer, criterion, device="cpu"):
    # save model
    now = datetime.now().strftime("%m%d_%H%M")
    save_model_root_path = getattr(cfg, "save_model_root_path", None)
    save_path = os.path.join(save_model_root_path, now + ".pth")
    
    # early stopping settings
    patience = getattr(cfg.exp_settings, "ealry_stopping_patience", 5)
    min_delta = getattr(cfg.exp_settings, "early_stopping_min_delta", 0.0)
    early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)
    
    # exp settings
    num_epochs = cfg.exp_settings.num_epochs
    best_val_loss = float('inf')


    # train & valid loop
    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for data in tqdm(dataloader['train'], desc=f"Epoch {epoch+1}/{num_epochs} Training"):
            x1, x2, target = data["x1"].to(device), data["x2"].to(device), data["y"].to(device)
            
            optimizer.zero_grad()
            out = model(x1, x2)
            logits = out.squeeze(1)
            
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            
            bs = target.size(0)
            train_loss_sum += loss.item() * bs
            train_count += bs

        train_loss = train_loss_sum / max(train_count, 1)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
        
        # validation
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        all_outputs = []
        all_targets = []        
        with torch.no_grad():
            for data in tqdm(dataloader['val'], desc=f"Epoch {epoch+1}/{num_epochs} Validation"):
                x1, x2, target = data["x1"].to(device), data["x2"].to(device), data["y"].to(device)

                out = model(x1, x2)
                logits = out.squeeze(1)
                val_loss = criterion(logits, target)
                
                bs = target.size(0)
                val_loss_sum += val_loss.item() * bs
                val_count += bs
                
                probs = torch.sigmoid(logits)

                all_outputs.append(probs)
                all_targets.append(target)
                
        val_loss = val_loss_sum / max(val_count, 1)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        roc_auc, accuracy, f1, sensitivity, specificity = evaluate_model(all_outputs, all_targets)

        print(f"Validation Loss: {val_loss:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, "
              f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
        print("")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            if getattr(args, "save_model", False):
                torch.save(model.state_dict(), save_path)
                print("Best model saved.")

        # # Logging
        # log_path = log_root_path, f"{now}_log.txt"
        # with open(log_path, "a") as f:
        #     f.write(f"{now} | Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n")
        
        if early_stopper.step(val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}. (patience={patience})")
            break        