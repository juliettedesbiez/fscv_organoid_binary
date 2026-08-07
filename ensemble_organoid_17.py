"""
Soft-voting ensemble for organoid BINARY classifier.
Averages class probabilities from RF, XGB (17-feature) and MLP then computes metrics.

Classes: 0=baseline (No Event), 1=spontaneous (Event)

Output is written to results_organoid_17/
"""

import os, json, numpy as np, pandas as pd
from sklearn.metrics import (f1_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)

BASE = r"C:\Users\julie\OneDrive - Imperial College London\organoid data output"
CLASS_NAMES = ['No Event', 'Event']

def compute_binary_metrics(y_true, y_proba):
    """Compute metrics for binary soft-voting output."""
    y_pred = np.argmax(y_proba, axis=1)

    metrics = {
        'f1_macro':    float(f1_score(y_true, y_pred, average='macro',    zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_per_class': {
            CLASS_NAMES[i]: float(f1_score(y_true, y_pred, labels=[i], average='micro', zero_division=0))
            for i in range(2)
        },
        'precision_macro': float(precision_score(y_true, y_pred, average='macro',    zero_division=0)),
        'recall_macro':    float(recall_score(y_true, y_pred, average='macro',       zero_division=0)),
        'auc': float(roc_auc_score(y_true, y_proba[:, 1])),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    }
    return metrics

def print_metrics(metrics, name):
    print(f"\n{name} ENSEMBLE RESULTS")
    print(f"  F1 Macro:    {metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
    print(f"  AUC:         {metrics['auc']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
    print("\n  Per-class F1:")
    for cls, f1 in metrics['f1_per_class'].items():
        print(f"    {cls}: {f1:.4f}")
    print("\n  Confusion Matrix (rows=true, cols=pred):")
    print(f"  {'':12}", "  ".join(f"{c:>12}" for c in CLASS_NAMES))
    for i, row in enumerate(metrics['confusion_matrix']):
        print(f"  {CLASS_NAMES[i]:12}", "  ".join(f"{v:>12}" for v in row))

def main():
    print("=" * 50)
    print("SOFT-VOTING ENSEMBLE (ORGANOID BINARY, 17-FEATURE RF/XGB)")
    print("=" * 50)

    # Load true labels from test set (17-feature version)
    test_meta = pd.read_csv(rf"{BASE}\windows_metadata_test_organoid_17.csv")
    y_test = test_meta['label'].values

    print(f"\nTest set: {len(y_test)} samples")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {(y_test == i).sum()}")

    # RF/XGB from the 17-feature retrain; MLP from the original (unretrained) run
    proba_paths = {
        'rf':  rf"{BASE}\results_organoid_17\rf_proba.npy",
        'xgb': rf"{BASE}\results_organoid_17\xgb_proba.npy",
        'mlp': rf"{BASE}\results_organoid_17\mlp_proba.npy",
    }

    missing = []
    probas = {}
    for model, path in proba_paths.items():
        if not os.path.exists(path):
            missing.append(model)
        else:
            probas[model] = np.load(path)
            assert probas[model].shape == (len(y_test), 2), \
                f"{model}_proba.npy shape {probas[model].shape} expected ({len(y_test)}, 2)"

    if missing:
        print(f"\nMissing probability files for: {missing}")
        return

    # Soft vote: average probabilities across models
    ensemble_proba = np.mean(list(probas.values()), axis=0)  # (n_samples, 2)

    metrics = compute_binary_metrics(y_test, ensemble_proba)
    print_metrics(metrics, "SOFT-VOTING")

    # Save to results_organoid_17
    os.makedirs(rf"{BASE}\results_organoid_17", exist_ok=True)
    np.save(rf"{BASE}\results_organoid_17\ensemble_proba.npy", ensemble_proba)
    json.dump(metrics, open(rf"{BASE}\results_organoid_17\ensemble_test.json", 'w'), default=float)
    print(f"\n✓ Saved results_organoid_17/ensemble_test.json and results_organoid_17/ensemble_proba.npy")

if __name__ == "__main__":
    main()
