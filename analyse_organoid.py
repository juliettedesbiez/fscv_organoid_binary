"""
analyse_organoid.py
Analysis and figure generation for organoid BINARY classifier.

Generates:
  figures_organoid/roc_curves_test.jpg
  figures_organoid/pr_curves_test.jpg
  figures_organoid/confusion_matrices_test.jpg
  figures_organoid/mlp_saliency.jpg
  figures_organoid/feature_importance_rf.jpg
  figures_organoid/feature_importance_xgb.jpg

Run after test_models_organoid.py.
Usage: python analyse_organoid.py [--config fscv_config_organoid.yaml]
"""

import os, json, pickle, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
import shap
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    ConfusionMatrixDisplay, matthews_corrcoef, roc_auc_score, f1_score
)
from sklearn.inspection import permutation_importance
from scipy.stats import skew, kurtosis
from extract_features_organoid import extract, load_config

warnings.filterwarnings('ignore')

# ── FONT CONFIGURATION ────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

BASE = r"C:\Users\julie\OneDrive - Imperial College London\organoid data output"
os.makedirs(rf"{BASE}\figures_organoid", exist_ok=True)

CLASS_NAMES  = ['No Event', 'Event']
COLORS       = {'RF': '#1f77b4', 'XGB': '#ff7f0e', 'MLP': '#2ca02c', 'Ensemble': '#d62728'}
FEAT_NAMES   = [
    'Peak Current', 'Peak Voltage', 'Peak Width (FWHM)', 'Trough Current',
    'AUC (Oxidation)', 'AUC (Full)', 'Mean', 'Std Dev',
    'Skewness', 'Kurtosis', 'Temporal Slope', 'Time of Max',
    'Rise Time', 'Decay Time', 'Ox/Red Ratio', 'Rise Slope', 'Ox/Red Lag'
]

# ── CONFIG ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='fscv_config_organoid.yaml')
args = parser.parse_args()
cfg = load_config(args.config)

with open(args.config) as f:
    _yaml = yaml.safe_load(f)
WINDOW_FRAMES = int(2.0 * _yaml['fscv_hz'])
N_VOLTAGE_PTS = 1100
MLP_INPUT     = N_VOLTAGE_PTS * WINDOW_FRAMES

# ── MLP ARCHITECTURE (must match train_models_organoid.py) ───────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(MLP_INPUT, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64),        nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2))          # binary output
    def forward(self, x): return self.net(x)

# ── LOAD TEST DATA (17-feature split — this is what RF/XGB were trained/tested on) ──
test_meta = pd.read_csv(rf"{BASE}\windows_metadata_test_organoid.csv")
y_test    = test_meta['label'].values
print(f"Test set: {len(y_test)} samples")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name} ({i}): {(y_test == i).sum()}")

# Extract features (consistent with training — extract(arr, cfg), 17 features)
print("\nExtracting features...")
feat_dicts = [extract(np.load(rf"{BASE}\window_arrays\{wid}.npy"), cfg)
              for wid in test_meta['window_id'].values]
feat_keys  = list(feat_dicts[0].keys())
X_feat     = np.array([[d[k] for k in feat_keys] for d in feat_dicts], dtype=np.float32)

X_raw = np.array([np.load(rf"{BASE}\window_arrays\{wid}.npy").flatten()
                  for wid in test_meta['window_id'].values], dtype=np.float32)

# ── LOAD MODELS AND GET PROBABILITIES ─────────────────────────────────────────
print("Loading models...")

# RF/XGB: 17-feature retrain
rf_data  = pickle.load(open(rf"{BASE}\models_organoid\rf_model.pkl",  'rb'))
xgb_data = pickle.load(open(rf"{BASE}\models_organoid\xgb_model.pkl", 'rb'))
# MLP: original run, unaffected by the feature-engineering change
mlp_data = pickle.load(open(rf"{BASE}\models_organoid\mlp_model.pkl", 'rb'))

rf_proba  = rf_data['model'].predict_proba(X_feat)        # (n, 2)
xgb_proba = xgb_data['model'].predict_proba(X_feat)       # (n, 2)

mlp_m = MLP()
mlp_m.load_state_dict(mlp_data['model_state'])
mlp_m.eval()
X_norm = (X_raw - mlp_data['mean']) / (mlp_data['std'] + 1e-8)
with torch.no_grad():
    mlp_proba = torch.softmax(mlp_m(torch.FloatTensor(X_norm)), dim=1).numpy()  # (n, 2)

ens_proba = (rf_proba + xgb_proba + mlp_proba) / 3.0

probas = {'RF': rf_proba, 'XGB': xgb_proba, 'MLP': mlp_proba, 'Ensemble': ens_proba}

# One-hot targets for OvR-style ROC/PR loop below (label_binarize collapses to
# a single column for binary input, so build the (n,2) array explicitly instead)
y_bin = np.column_stack([1 - y_test, y_test])   # (n, 2)

# ── SUMMARY METRICS TO TERMINAL ───────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"{'Model':<12} {'F1_macro':>10} {'MCC':>8} {'AUC':>10}")
print("=" * 55)
for name, proba in probas.items():
    preds = np.argmax(proba, axis=1)
    f1    = f1_score(y_test, preds, average='macro', zero_division=0)
    mcc   = matthews_corrcoef(y_test, preds)
    auc_score = roc_auc_score(y_test, proba[:, 1])   # binary AUC — positive class column
    print(f"{name:<12} {f1:>10.4f} {mcc:>8.4f} {auc_score:>10.4f}")
print("=" * 55)

# ── 1. ROC CURVES (per class + macro, one plot per model) ─────────────────────
print("\n[1/6] ROC Curves...")
fig, axes = plt.subplots(1, 4, figsize=(22, 6), dpi=200)
for ax, (name, proba) in zip(axes, probas.items()):
    for i, cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        ax.plot(fpr, tpr, lw=2, label=f'{cls} (AUC={auc(fpr, tpr):.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('FPR', fontsize=10)
    ax.set_ylabel('TPR', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.suptitle('ROC Curves — Test Set', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(rf"{BASE}\figures_organoid\roc_curves_test.jpg", dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ roc_curves_test.jpg")

# ── 2. PRECISION-RECALL CURVES ────────────────────────────────────────────────
print("[2/6] Precision-Recall Curves...")
fig, axes = plt.subplots(1, 4, figsize=(22, 6), dpi=200)
for ax, (name, proba) in zip(axes, probas.items()):
    for i, cls in enumerate(CLASS_NAMES):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], proba[:, i])
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, lw=2, label=f'{cls} (AUC={pr_auc:.3f})')
        chance = y_bin[:, i].mean()
        ax.axhline(chance, linestyle=':', lw=1, alpha=0.4)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Recall', fontsize=10)
    ax.set_ylabel('Precision', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.suptitle('Precision-Recall Curves — Test Set', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(rf"{BASE}\figures_organoid\pr_curves_test.jpg", dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ pr_curves_test.jpg")

# ── 3. CONFUSION MATRICES (2x2 grid, binary) ──────────────────────────────────
print("[3/6] Confusion Matrices...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=200)
for idx, (name, proba) in enumerate(probas.items()):
    preds = np.argmax(proba, axis=1)
    cm    = confusion_matrix(y_test, preds, labels=[0, 1])
    mcc   = matthews_corrcoef(y_test, preds)
    disp  = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=axes.flat[idx], cmap='Blues', colorbar=False)
    axes.flat[idx].set_title(f'{name}  (MCC={mcc:.3f})', fontsize=12, fontweight='bold')
    axes.flat[idx].tick_params(axis='x', rotation=30)
plt.suptitle('Confusion Matrices — Test Set', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(rf"{BASE}\figures_organoid\confusion_matrices_test.jpg", dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ confusion_matrices_test.jpg")

# ── 4. MLP GRADIENT SALIENCY ──────────────────────────────────────────────────
print("[4/6] MLP Gradient Saliency...")
mlp_m.eval()
X_sal = torch.FloatTensor(X_norm)
X_sal.requires_grad_(True)

# Sum logits across both classes for class-agnostic saliency
logits = mlp_m(X_sal).sum()
logits.backward()

saliency    = np.abs(X_sal.grad.detach().numpy()).mean(axis=0)
saliency_2d = saliency.reshape(N_VOLTAGE_PTS, WINDOW_FRAMES)

fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
im = ax.imshow(saliency_2d, aspect='auto', cmap='hot', origin='lower')
plt.colorbar(im, ax=ax, label='Mean |Gradient|')
ax.set_xlabel(f'Time Frames (×{1/cfg["fscv_hz"]:.1f}s each)', fontsize=12)
ax.set_ylabel('Voltage Points', fontsize=10)
ax.set_title('MLP Gradient Saliency Map — Input Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(rf"{BASE}\figures_organoid\mlp_saliency.jpg", dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ mlp_saliency.jpg")

# ── 5A. RANDOM FOREST — PERMUTATION IMPORTANCE ────────────────────────────────
print("[5a/6] RF Permutation Importance...")
rf_perm    = permutation_importance(rf_data['model'], X_feat, y_test,
                                    n_repeats=10, random_state=42, n_jobs=-1)
rf_imp     = rf_perm.importances_mean
rf_imp_norm = rf_imp / (rf_imp.max() + 1e-12)

rf_df = pd.DataFrame({'Feature': FEAT_NAMES, 'Importance': rf_imp_norm})\
          .sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
ax.barh(rf_df['Feature'], rf_df['Importance'], color=COLORS['RF'])
ax.set_xlabel('Normalised Permutation Importance (max=1)', fontsize=12)
ax.set_title('Random Forest — Permutation Feature Importance (17 features)', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(rf"{BASE}\figures_organoid\feature_importance_rf.jpg", dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ feature_importance_rf.jpg")

# ── 5B. XGBOOST — SHAP ────────────────────────────────────────────────────────
print("[5b/6] XGBoost SHAP...")
explainer  = shap.TreeExplainer(xgb_data['model'])
shap_vals  = explainer.shap_values(X_feat)

if isinstance(shap_vals, list):
    shap_arr = np.stack(shap_vals, axis=0)
    xgb_imp  = np.abs(shap_arr).mean(axis=(0, 1))
else:
    xgb_imp  = np.abs(shap_vals).mean(axis=0)

xgb_imp_norm = xgb_imp / (xgb_imp.sum() + 1e-12)

xgb_df = pd.DataFrame({'Feature': FEAT_NAMES, 'Importance': xgb_imp_norm})\
           .sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
ax.barh(xgb_df['Feature'], xgb_df['Importance'], color=COLORS['XGB'])
ax.set_xlabel('Mean |SHAP| (normalised)', fontsize=12)
ax.set_title('XGBoost — SHAP Feature Importance (Binary, 17 features)', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(rf"{BASE}\figures_organoid\feature_importance_xgb.jpg", dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ feature_importance_xgb.jpg")

print("\n✓ ANALYSIS COMPLETE — 6 figures saved to figures_organoid/")
