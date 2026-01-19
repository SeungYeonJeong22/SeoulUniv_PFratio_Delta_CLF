from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
import torch


def evaluate_model(probs, targets, threshold=0.5):
    probs = probs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    targets = targets.astype(int)
    preds = (probs >= threshold).astype(int)
    
    roc_auc = roc_auc_score(targets, preds)
    accuracy = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds)
    
    cm = confusion_matrix(targets, preds, labels=[0, 2])
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return roc_auc, accuracy, f1, sensitivity, specificity