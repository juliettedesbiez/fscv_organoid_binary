"""Shared utilities for FSCV classification pipeline — Organoid (BINARY)."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42
CLASS_NAMES  = ['No Event', 'Event']

# BASE covers everything this file reads — features_organoid_17.csv and window_arrays/ both live under it
BASE = r"C:\Users\julie\OneDrive - Imperial College London\organoid data output"


def load_features():
    """Load engineered features from features_organoid_17.csv."""
    print("Loading engineered features...")
    df = pd.read_csv(rf"{BASE}\features_organoid_17.csv")
    feat_cols = [c for c in df.columns if c not in ['window_id', 'label', 'group_id']]
    X = df[feat_cols].values.astype(np.float32)
    y, groups = df['label'].values, df['group_id'].astype(str).values
    print(f"  {len(y)} windows, {len(feat_cols)} features, {len(np.unique(groups))} groups")
    return X, y, groups


def load_raw_for_features():
    """Load raw flattened windows matching features_organoid_17.csv order."""
    print("Loading raw windows...")
    df = pd.read_csv(rf"{BASE}\features_organoid_17.csv")
    X = np.array([np.load(rf"{BASE}\window_arrays\{wid}.npy").flatten()
              for wid in df['window_id']], dtype=np.float32)
    y, groups = df['label'].values, df['group_id'].astype(str).values
    print(f"  {len(y)} windows, {X.shape[1]} raw features")
    return X, y, groups


def compute_metrics(y_true, y_proba):
    """
    Compute binary metrics from probability array (n_samples x 2).
    Returns per-class F1, macro F1, weighted F1, AUC, confusion matrix.
    """
    y_pred = np.argmax(y_proba, axis=1)
    metrics = {
        'f1_macro':    float(f1_score(y_true, y_pred, average='macro',    zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_per_class': {
            CLASS_NAMES[i]: float(f1_score(y_true, y_pred, labels=[i], average='micro', zero_division=0))
            for i in range(2)
        },
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro':    float(recall_score(y_true, y_pred, average='macro',    zero_division=0)),
        'auc': float(roc_auc_score(y_true, y_proba[:, 1])),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    }
    return metrics


def print_metrics(metrics, name):
    """Print binary metrics to terminal."""
    print(f"\n{name} RESULTS")
    print(f"  F1 Macro:    {metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
    print(f"  AUC:         {metrics['auc']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
    print("  Per-class F1:")
    for cls, f1 in metrics['f1_per_class'].items():
        print(f"    {cls}: {f1:.4f}")
    print("  Confusion Matrix (rows=true, cols=pred):")
    print(f"    {'':14}", "  ".join(f"{c:>14}" for c in CLASS_NAMES))
    for i, row in enumerate(metrics['confusion_matrix']):
        print(f"    {CLASS_NAMES[i]:14}", "  ".join(f"{v:>14}" for v in row))


def select_models(action="train"):
    """Interactive model selection."""
    print(f"\nSelect models to {action}: 1=RF  2=XGB  3=MLP  4=ALL  0=Exit")
    choice = input("Choice: ").strip()
    if choice == '0': return []
    if choice == '4': return ['rf', 'xgb', 'mlp']
    return [{'1': 'rf', '2': 'xgb', '3': 'mlp'}.get(c) for c in choice if c in '123']
