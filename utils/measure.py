from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score, confusion_matrix
# from torchmetrics.classification import BinarySpecificityAtSensitivity
import torch
import numpy as np


def evaluate_model(probs, targets):
    probs = probs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    targets = targets.astype(int)
    
        # 유든 인덱스 사용
    fpr, tpr, thresholds = roc_curve(targets, probs)
    youden_index = tpr - fpr
    best_idx = np.argmax(youden_index)
    best_threshold = thresholds[best_idx]
    
    preds = (probs >= best_threshold).astype(int)
        
    roc_auc = roc_auc_score(targets, probs)
    accuracy = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds)
    
    cm = confusion_matrix(targets, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return roc_auc, accuracy, f1, sensitivity, specificity, best_threshold