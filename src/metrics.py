from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def evaluate_binary(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0

    best_idx = int(np.argmax(2 * precision * recall / np.clip(precision + recall, 1e-9, None)))
    best_threshold = (
        float(thresholds[max(0, min(best_idx, len(thresholds) - 1))]) if len(thresholds) else 0.5
    )

    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "best_f1": float(2 * precision[best_idx] * recall[best_idx] / max(precision[best_idx] + recall[best_idx], 1e-9)),
        "best_threshold": best_threshold,
    }
