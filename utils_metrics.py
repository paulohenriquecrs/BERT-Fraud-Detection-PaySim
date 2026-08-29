"""Utility functions for evaluating fraud detection models."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def find_best_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    step: float = 0.05,
    min_t: float = 0.00,
    max_t: float = 0.99,
) -> tuple[float, float]:
    """Finds the decision threshold that maximizes F1-Score on evaluation/validation data."""
    best_f1 = -1.0
    best_threshold = 0.5

    for t in np.arange(min_t, max_t, step):
        preds = (y_probs >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, preds, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, best_f1


def evaluate_model_pipeline(
    y_eval_true: np.ndarray,
    y_eval_probs: np.ndarray,
    y_test_true: np.ndarray,
    y_test_probs: np.ndarray,
    threshold_step: float = 0.05,
):
    """1. Tunes optimal threshold on EVAL data.

    2. Evaluates performance on TEST data using tuned threshold. 3. Displays a
    clean structured report.
    """
    # --- Step 1: Tune Threshold on Evaluation Set ---
    optimal_threshold, eval_f1 = find_best_threshold(
        y_eval_true, y_eval_probs, step=threshold_step
    )

    # --- Step 2: Apply Threshold to Test Set ---
    y_test_pred = (y_test_probs >= optimal_threshold).astype(int)

    # Compute threshold-independent ranking metrics
    test_pr_auc = average_precision_score(y_test_true, y_test_probs)
    test_roc_auc = roc_auc_score(y_test_true, y_test_probs)

    # Compute threshold-dependent metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_true, y_test_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_test_true, y_test_pred).ravel()

    # --- Step 3: Print Formatted Report ---
    print("=" * 60)
    print("                  MODEL EVALUATION REPORT                   ")
    print("=" * 60)
    print(
        f"Selected Threshold (from Eval): {optimal_threshold:.2f}  (Eval F1:"
        f" {eval_f1:.4f})"
    )
    print("-" * 60)

    print("OVERALL PERFORMANCE (Test Set):")
    print(f"  • PR-AUC (Avg Precision) : {test_pr_auc:.4f}")
    print(f"  • ROC-AUC                : {test_roc_auc:.4f}")
    print(f"  • F1-Score               : {f1:.4f}")
    print(f"  • Precision              : {precision:.4f}")
    print(f"  • Recall                 : {recall:.4f}")
    print("-" * 60)

    print("CONFUSION MATRIX (Test Set):")
    print(f"  • True Negatives  (TN)   : {tn:,}")
    print(f"  • False Positives (FP)   : {fp:,}")
    print(f"  • False Negatives (FN)   : {fn:,}")
    print(f"  • True Positives  (TP)   : {tp:,}")
    print("-" * 60)

    print("DETAILED CLASSIFICATION REPORT (Test Set):")
    print(
        classification_report(
            y_test_true,
            y_test_pred,
            target_names=["Normal", "Fraud"],
            digits=4,
        )
    )
    print("=" * 60)

    return {
        "optimal_threshold": optimal_threshold,
        "test_f1": f1,
        "test_precision": precision,
        "test_recall": recall,
        "test_pr_auc": test_pr_auc,
        "test_roc_auc": test_roc_auc,
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
    }
