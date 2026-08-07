"""
Train RF, XGBoost, and MLP — organoid BINARY classifier.
Classes: 0=baseline (No Event), 1=spontaneous (Event)

Architecture/training loop ported from Phase 2's tuned MLP config
(lr=1e-4, ReduceLROnPlateau, 100 epochs, patience=15, class weight x4 on
minority class) — the config that produced Phase 2's best CV result
(macro F1=0.7764) — with the output layer/loss switched from 3-class to binary.

Usage: python train_models_organoid.py [all|rf|xgb|mlp]
Run extract_features_organoid.py first.
"""

import os, sys, pickle, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import xgboost as xgb
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils_organoid import load_features, load_raw_for_features, compute_metrics, print_metrics, select_models, RANDOM_STATE

import yaml

with open("fscv_config_organoid.yaml") as f:
    _cfg = yaml.safe_load(f)
WINDOW_FRAMES = int(2.0 * _cfg['fscv_hz'])   # 20 frames, matches make_windows_organoid.py
N_VOLTAGE_PTS = 1100
MLP_INPUT     = N_VOLTAGE_PTS * WINDOW_FRAMES

N_SPLITS = 5
BASE = r"C:\Users\julie\OneDrive - Imperial College London\organoid data output"

os.makedirs(rf"{BASE}\models_organoid_17", exist_ok=True)


def main(selected=None):
    args = [a.lower() for a in sys.argv[1:]]
    if args:
        selected = ['rf', 'xgb', 'mlp'] if 'all' in args else [a for a in args if a in ['rf', 'xgb', 'mlp']]
    elif selected is None:
        selected = select_models("train")

    if not selected: print("No models selected."); return
    if not os.path.exists(rf"{BASE}\features_organoid_17.csv"):
        print("features_organoid_17.csv not found — run extract_features_organoid_17.py first"); return

    X_feat, y, groups = load_features()
    X_raw, _, _ = load_raw_for_features()

    results = {}
    if 'rf'  in selected: results['rf']  = train_rf(X_feat, y, groups)
    if 'xgb' in selected: results['xgb'] = train_xgb(X_feat, y, groups)
    if 'mlp' in selected: results['mlp'] = train_mlp(X_raw,  y, groups)

    print("\n" + "="*40 + "\nSUMMARY\n" + "="*40)
    for n, m in results.items():
        print(f"  {n.upper()}: F1_macro={m['f1_macro']:.4f}  AUC={m['auc']:.4f}")


def train_rf(X, y, groups):
    print("\nTraining Random Forest (binary)...")
    gkf = GroupKFold(N_SPLITS)
    y_true_all, y_proba_all = [], []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        clf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                     class_weight='balanced', n_jobs=-1,
                                     random_state=RANDOM_STATE)
        clf.fit(X[tr], y[tr])
        y_true_all.extend(y[te])
        y_proba_all.extend(clf.predict_proba(X[te]))   # shape (n, 2) — sklearn infers this automatically
        print(f"  Fold {fold+1}/{N_SPLITS}")

    metrics = compute_metrics(np.array(y_true_all), np.array(y_proba_all))
    print_metrics(metrics, "RF")

    final = RandomForestClassifier(n_estimators=200, max_depth=20,
                                   class_weight='balanced', n_jobs=-1,
                                   random_state=RANDOM_STATE)
    final.fit(X, y)
    pickle.dump({'model': final}, open(rf"{BASE}\models_organoid_17\rf_model.pkl", 'wb'))
    return metrics


def train_xgb(X, y, groups):
    print("\nTraining XGBoost (binary)...")
    gkf = GroupKFold(N_SPLITS)
    y_true_all, y_proba_all = [], []

    # Per-class sample weights — same manual approach as Phase 2 (kept for consistency
    # with your established methodology, rather than switching to scale_pos_weight)
    class_counts = np.bincount(y)
    sample_weight = np.array([1.0 / class_counts[c] for c in y])
    sample_weight = sample_weight / sample_weight.mean()

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        clf = xgb.XGBClassifier(n_estimators=200, max_depth=6,
                                 objective='binary:logistic',
                                 random_state=RANDOM_STATE, verbosity=0)
        clf.fit(X[tr], y[tr], sample_weight=sample_weight[tr])
        y_true_all.extend(y[te])
        y_proba_all.extend(clf.predict_proba(X[te]))   # shape (n, 2)
        print(f"  Fold {fold+1}/{N_SPLITS}")

    metrics = compute_metrics(np.array(y_true_all), np.array(y_proba_all))
    print_metrics(metrics, "XGB")

    final = xgb.XGBClassifier(n_estimators=200, max_depth=6,
                               objective='binary:logistic',
                               random_state=RANDOM_STATE, verbosity=0)
    final.fit(X, y, sample_weight=sample_weight)
    pickle.dump({'model': final}, open(rf"{BASE}\models_organoid_17\xgb_model.pkl", 'wb'))
    return metrics


def train_mlp(X, y, groups):
    print(f"\nTraining MLP (binary, input={MLP_INPUT})...")

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(MLP_INPUT, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 64),        nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, 2))          # 2 output classes (was 3 in Phase 2)
        def forward(self, x): return self.net(x)

    # Class weights: minority class (1 = spontaneous/Event) boosted x3,
    # matching your Phase 2 BEST config (v7: weighted CE + WeightedRandomSampler stacked,
    # confirmed as the winning arm of your 4-way loss/sampler comparison)
    class_counts = np.bincount(y)
    class_weights = torch.FloatTensor(1.0 / class_counts)
    class_weights[1] *= 3.0
    class_weights = class_weights / class_weights.sum()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    gkf = GroupKFold(N_SPLITS)
    y_true_all, y_proba_all = [], []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        X_tr_mean = X[tr].mean(); X_tr_std = X[tr].std() + 1e-8
        X_tr = (X[tr] - X_tr_mean) / X_tr_std
        X_te = (X[te] - X_tr_mean) / X_tr_std

        X_tr_t = torch.FloatTensor(X_tr)
        y_tr_t = torch.LongTensor(y[tr])
        X_te_t = torch.FloatTensor(X_te)

        # Sampler stacked with weighted loss, matching v7's winning combo
        tr_class_counts = np.bincount(y[tr])
        sample_weights = 1.0 / tr_class_counts[y[tr]]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )

        loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, sampler=sampler)

        model = MLP()
        opt    = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)

        best_f1, patience, best_proba = -1.0, 0, None

        for epoch in range(100):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                criterion(model(xb), yb).backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                logits = model(X_te_t)
                proba  = torch.softmax(logits, dim=1).numpy()  # (n, 2)
                preds  = np.argmax(proba, axis=1)
                f1     = f1_score(y[te], preds, average='macro', zero_division=0)

                print(f"    Fold {fold+1}/{N_SPLITS} Epoch {epoch+1:2d}/100 | F1_macro: {f1:.4f}")
                sched.step(f1)

                if f1 > best_f1:
                    best_f1, best_proba, patience = f1, proba.copy(), 0
                else:
                    patience += 1
                    if patience >= 15:
                        print(f"    Early stop at epoch {epoch+1}"); break

        y_true_all.extend(y[te])
        y_proba_all.extend(best_proba)
        print(f"  Fold {fold+1}/{N_SPLITS} best F1_macro={best_f1:.4f}")

    np.save(rf"{BASE}\models_organoid_17\mlp_oof_ytrue.npy", np.array(y_true_all))
    np.save(rf"{BASE}\models_organoid_17\mlp_oof_yproba.npy", np.array(y_proba_all))

    metrics = compute_metrics(np.array(y_true_all), np.array(y_proba_all))
    print_metrics(metrics, "MLP")

    # Final model trained on all data
    X_all_mean = X.mean(); X_all_std = X.std() + 1e-8
    X_all = (X - X_all_mean) / X_all_std

    all_class_counts = np.bincount(y)
    all_sample_weights = 1.0 / all_class_counts[y]
    all_sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.DoubleTensor(all_sample_weights),
        num_samples=len(all_sample_weights),
        replacement=True
    )
    final  = MLP()
    opt    = torch.optim.Adam(final.parameters(), lr=1e-4, weight_decay=1e-5)
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_all),
                                      torch.LongTensor(y)), batch_size=32, sampler=all_sampler)
    for _ in range(30):
        final.train()
        for xb, yb in loader:
            opt.zero_grad(); criterion(final(xb), yb).backward(); opt.step()

    pickle.dump({'model_state': final.state_dict(),
                 'mean': X_all_mean, 'std': X_all_std},
                 open(rf"{BASE}\models_organoid_17\mlp_model.pkl", 'wb'))
    return metrics


if __name__ == "__main__":
    main()
