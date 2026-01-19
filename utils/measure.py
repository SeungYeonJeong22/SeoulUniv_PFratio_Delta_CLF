from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
import torch


def evaluate_model(output, target, device="cpu"):
    roc_auc_score = roc_auc_score(target.cpu().numpy(), torch.sigmoid(output).cpu().numpy())
    accuracy = accuracy_score(target.cpu().numpy(), (torch.sigmoid(output).cpu().numpy() > 0.5).astype(int))
    f1 = f1_score(target.cpu().numpy(), (torch.sigmoid(output).cpu().numpy() > 0.5).astype(int))
    tn, fp, fn, tp = confusion_matrix(target.cpu().numpy(), (torch.sigmoid(output).cpu().numpy() > 0.5).astype(int)).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return roc_auc_score, accuracy, f1, sensitivity, specificity