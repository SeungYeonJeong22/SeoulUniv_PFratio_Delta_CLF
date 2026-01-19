import torch
from tqdm import tqdm
from datetime import datetime
import os
from utils.measure import evaluate_model

def train(args, cfg, dataloader, model, optimizer, criterion, device="cpu"):
    now = datetime.now().strftime("%m%d_%H%M")
    save_model_root_path = cfg.save_model_root_path
    log_root_path = cfg.log_root_path
    
    num_epochs = args.num_epochs
    best_val_loss = float('inf')
    
    save_path = os.path.join(save_model_root_path + now + ".pth")
    

    # train & valid loop
    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for (x1, x2), target in tqdm(dataloader['train'], desc=f"Epoch {epoch+1}/{num_epochs} Training"):
            x1, x2, target = x1.to(device), x2.to(device), target.to(device)
            
            optimizer.zero_grad()
            logits = model(x1, x2).squeeze(1)
            
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
            for (x1, x2), target in tqdm(dataloader['val'], desc=f"Epoch {epoch+1}/{num_epochs} Validation"):
                x1, x2, target = x1.to(device), x2.to(device), target.to(device)

                logits = model(x1, x2).squeeze(1)
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

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            torch.save(model.state_dict(), save_path)
            print("Best model saved.")

        # # Logging
        # log_path = log_root_path, f"{now}_log.txt"
        # with open(log_path, "a") as f:
        #     f.write(f"{now} | Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n")