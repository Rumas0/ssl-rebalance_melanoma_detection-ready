import time
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, precision_recall_curve, roc_auc_score, average_precision_score,
)


def evaluate(model, loader, device):
    """Returns (accuracy, macro_f1, labels, preds, probs)."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.append(probs)
    all_probs = np.concatenate(all_probs, axis=0)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, macro_f1, all_labels, all_preds, all_probs


def measure_latency(model, loader, device, n_batches=10):
    model.eval()
    times = []
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= n_batches:
                break
            images = images.to(device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed / images.size(0))
    return float(np.mean(times) * 1000) if times else None


def compute_full_report(true_labels, pred_labels, probs, classes):
    """
    Compute the full metric suite the manuscript needs.

    Returns a dict with:
      - accuracy, macro_f1
      - per-class recall/precision/f1
      - MEL one-vs-rest ROC-AUC and PR-AUC (average precision)
      - confusion matrix
    """
    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    report = classification_report(
        true_labels, pred_labels, target_names=classes,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(true_labels, pred_labels)

    mel_idx = classes.index('MEL')
    binary_true = (np.array(true_labels) == mel_idx).astype(int)
    mel_scores = probs[:, mel_idx]

    roc_auc = float(roc_auc_score(binary_true, mel_scores)) if binary_true.sum() > 0 else None
    pr_auc = float(average_precision_score(binary_true, mel_scores)) if binary_true.sum() > 0 else None
    precision, recall, _ = precision_recall_curve(binary_true, mel_scores)

    return {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'mel_recall': float(report['MEL']['recall']),
        'mel_precision': float(report['MEL']['precision']),
        'mel_f1': float(report['MEL']['f1-score']),
        'bkl_recall': float(report['BKL']['recall']),
        'nv_recall': float(report['NV']['recall']),
        'mel_roc_auc': roc_auc,
        'mel_pr_auc': pr_auc,
        'confusion_matrix': cm.tolist(),
        'report': report,
        'pr_curve': {'precision': precision.tolist(), 'recall': recall.tolist()},
    }