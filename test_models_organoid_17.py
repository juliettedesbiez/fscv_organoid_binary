"""
Test trained models on held-out test set (run train_models_organoid.py first).
Classes: 0=baseline (No Event), 1=spontaneous (Event)

Usage: python test_models_organoid.py [all|rf|xgb|mlp] [--config fscv_config_organoid.yaml]
"""

import os, sys, json, pickle, argparse
import numpy as np, pandas as pd
import torch, torch.nn as nn
from utils_organoid import compute_metrics, print_metrics, select_models
from extract_features_organoid_17 import extract, load_config

import yaml

with open("fscv_config_organoid.yaml") as f:
    _cfg = yaml.safe_load(f)
WINDOW_FRAMES = int(2.0 * _cfg['fscv_hz'])
N_VOLTAGE_PTS = 1100
MLP_INPUT     = N_VOLTAGE_PTS * WINDOW_FRAMES

BASE = r"C:\Users\julie\OneDrive - Imperial College London\organoid data output retrain 3"

os.makedirs(rf"{BASE}\results_organoid_17", exist_ok=True)

def test_rf(X, y):
    path = rf"{BASE}\models_organoid_17\rf_model.pkl"
    if not os.path.exists(path): print("RF model not found"); return None
    data  = pickle.load(open(path, 'rb'))
    proba = data['model'].predict_proba(X)         # (n, 2)
    metrics = compute_metrics(y, proba)
    print_metrics(metrics, 'RF')
    json.dump(metrics, open(rf"{BASE}\results_organoid_17\rf_test.json", 'w'), default=float)
    np.save(rf"{BASE}\results_organoid_17\rf_proba.npy", proba)
    return metrics

def test_xgb(X, y):
    path = rf"{BASE}\models_organoid_17\xgb_model.pkl"
    if not os.path.exists(path): print("XGB model not found"); return None
    data  = pickle.load(open(path, 'rb'))
    proba = data['model'].predict_proba(X)         # (n, 2)
    metrics = compute_metrics(y, proba)
    print_metrics(metrics, 'XGB')
    json.dump(metrics, open(rf"{BASE}\results_organoid_17\xgb_test.json", 'w'), default=float)
    np.save(rf"{BASE}\results_organoid_17\xgb_proba.npy", proba)
    return metrics

def test_mlp(X, y):
    path = rf"{BASE}\models_organoid_17\mlp_model.pkl"
    if not os.path.exists(path): print("MLP model not found"); return None

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(MLP_INPUT, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 64),        nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, 2))
        def forward(self, x): return self.net(x)

    data   = pickle.load(open(path, 'rb'))
    model  = MLP()
    model.load_state_dict(data['model_state']); model.eval()

    X_norm = (X - data['mean']) / (data['std'] + 1e-8)
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_norm))
        proba  = torch.softmax(logits, dim=1).numpy()   # (n, 2)

    metrics = compute_metrics(y, proba)
    print_metrics(metrics, 'MLP')
    json.dump(metrics, open(rf"{BASE}\results_organoid_17\mlp_test.json", 'w'), default=float)
    np.save(rf"{BASE}\results_organoid_17\mlp_proba.npy", proba)
    return metrics

def main(selected=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='fscv_config_organoid.yaml')
    parser.add_argument('models', nargs='*')
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.models:
        selected = ['rf', 'xgb', 'mlp'] if 'all' in args.models else \
                   [m for m in args.models if m in ['rf', 'xgb', 'mlp']]
    elif selected is None:
        selected = select_models("test")
    if not selected: return

    print("\nLoading TEST set...")
    test_meta = pd.read_csv(rf"{BASE}\windows_metadata_test_organoid_17.csv")
    y_test    = test_meta['label'].values

    print(f"Test set: {len(y_test)} samples")
    print(f"  Baseline (0):    {(y_test==0).sum()}")
    print(f"  Spontaneous (1): {(y_test==1).sum()}")

    # Extract features using same function as training (from extract_features_organoid.py)
    print("\nExtracting features from test windows...")
    X_feat = np.array([extract(np.load(rf"{BASE}\window_arrays\{wid}.npy"), cfg)
                       for wid in test_meta['window_id'].values])
    feat_keys = list(X_feat[0].keys())
    X_feat = np.array([[row[k] for k in feat_keys] for row in X_feat], dtype=np.float32)

    X_raw  = np.array([np.load(rf"{BASE}\window_arrays\{wid}.npy").flatten()
                       for wid in test_meta['window_id'].values], dtype=np.float32)

    print(f"Features: {X_feat.shape}  Raw: {X_raw.shape}\n")

    results = {}
    if 'rf'  in selected: results['rf']  = test_rf(X_feat, y_test)
    if 'xgb' in selected: results['xgb'] = test_xgb(X_feat, y_test)
    if 'mlp' in selected: results['mlp'] = test_mlp(X_raw,  y_test)

    print("\n" + "="*40)
    for name, m in results.items():
        if m: print(f"{name.upper()}: F1_macro={m['f1_macro']:.4f}  AUC={m['auc']:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
