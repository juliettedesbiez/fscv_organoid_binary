"""
Extract features (12 original + rise_time, decay_time [fixed], ox_red_ratio
[fixed], rise_slope, ox_red_lag = 17 features) then create balanced 70:30
train/test split (group-aware, no data leakage) — Organoid (BINARY).
 
decay_time and ox_red_ratio use the FIXED formulas from the 3-class 28-feature
round (extract_features_3class_28.py) — the earlier 15-feature versions of
these two were flagged as unstable (decay_time collapses near-zero for real
events; ox_red_ratio has extreme outliers) and are NOT used here.
 
ox_red_lag is the key addition for organoid: it directly encodes the gap
between the oxidation peak and the reduction trough, i.e. the paired
oxidation/reduction relationship the interpretability gate is checking for.
Previously, trough_current was (incorrectly) computed from the oxidation
slice — RF/XGB had no reduction-band-specific feature at all. Fixed here.
 
Usage: python extract_features_organoid.py [--config fscv_config_organoid.yaml]
 
Classes: 0=baseline, 1=spontaneous
"""
 
import argparse
import numpy as np
import pandas as pd
import yaml
from scipy.stats import skew, kurtosis
from utils_organoid import RANDOM_STATE
 
BASE = r"C:\Users\julie\OneDrive - Imperial College London\organoid data output retrain 3"
 
def load_config(path="fscv_config_organoid.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
 
def extract(arr, cfg):
    """Extract 17 features from a window array using config voltage indices."""
    v0 = cfg['v_oxidation_start']
    v1 = cfg['v_oxidation_end']
    r0 = cfg['v_reduction_start']
    r1 = cfg['v_reduction_end']
 
    ox   = arr[v0:v1, :]   # oxidation region
    red  = arr[r0:r1, :]   # reduction region
    flat = arr.flatten()
 
    ox_trace  = ox.mean(axis=0)
    red_trace = red.mean(axis=0)
    n_frames  = len(ox_trace)
 
    peak_frame = ox_trace.argmax()
    peak_val   = ox_trace[peak_frame]
    red_frame  = red_trace.argmin()
 
    rise_time  = peak_frame
 
    # decay_time: fixed formula (rate of decline per frame after peak, not
    # "first frame below half-max" which collapses near-zero for real events)
    frames_left = n_frames - 1 - peak_frame
    decay_time  = (peak_val - ox_trace[-1]) / frames_left if frames_left > 0 else 0.0
 
    rise_slope  = (peak_val - ox_trace[0]) / max(peak_frame, 1)
 
    # ox_red_lag: frames between oxidation peak and reduction trough —
    # directly encodes the paired oxidation/reduction relationship
    ox_red_lag  = red_frame - peak_frame
 
    # ox_red_ratio: fixed formula (larger epsilon + clipped, avoids the
    # extreme-outlier blowups the original 15-feature version had)
    raw_ratio    = ox.max() / (abs(red.min()) + 1e-3)
    ox_red_ratio = float(np.clip(raw_ratio, -10, 10))
 
    return {
        'peak_current':   ox.max(),
        'peak_voltage':   ox.mean(axis=1).argmax() + v0,
        'peak_width':     (ox.mean(axis=1) > ox.mean(axis=1).max() * 0.5).sum(),
        'trough_current': red.min(),          # fixed: uses reduction region, not oxidation
        'auc_sero':       np.abs(ox).sum(),
        'auc_full':       np.abs(arr).sum(),
        'mean':           flat.mean(),
        'std':            flat.std(),
        'skewness':       skew(flat),
        'kurtosis':       kurtosis(flat),
        'time_change':    arr.mean(axis=0)[-5:].mean() - arr.mean(axis=0)[:5].mean(),
        'time_of_max':    arr.mean(axis=0).argmax(),
        'rise_time':      rise_time,
        'decay_time':     decay_time,
        'ox_red_ratio':   ox_red_ratio,
        'rise_slope':     rise_slope,
        'ox_red_lag':     ox_red_lag,
    }
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='fscv_config_organoid.yaml')
    args = parser.parse_args()
 
    cfg = load_config(args.config)
    balance_ratio = cfg['balance_ratio']
 
    print("Loading ALL windows...")
    meta = pd.read_csv(rf"{BASE}\windows_metadata.csv")
 
    print(f"Total: {len(meta)} windows")
    print(f"  Spontaneous: {(meta['label']==1).sum()}")
    print(f"  Baseline: {(meta['label']==0).sum()}")
 
    # BALANCE: Keep all spontaneous + balance_ratio× random baseline
    print(f"\nBalancing: all spontaneous + {balance_ratio}× random baseline...")
    spont = meta[meta['label'] == 1]
    baseline = meta[meta['label'] == 0]
 
    np.random.seed(RANDOM_STATE)
    baseline = baseline.sample(min(len(baseline), balance_ratio * len(spont)), random_state=RANDOM_STATE)
    balanced = pd.concat([spont, baseline]).reset_index(drop=True)
 
    print(f"Balanced set: {len(balanced)} windows")
    print(f"  Spontaneous: {(balanced['label']==1).sum()}")
    print(f"  Baseline: {(balanced['label']==0).sum()}")
    print(f"  Ratio: {(balanced['label']==0).sum()/(balanced['label']==1).sum():.1f}:1")
 
    # SPLIT: 70/30 on balanced data, GROUP-AWARE (whole organoids go to train or test, never split)
    print("\nSplitting 70% train / 30% test (group-aware)...")
    np.random.seed(RANDOM_STATE)
    groups = balanced['group_id'].unique()
    np.random.shuffle(groups)
    n_test_groups = max(1, int(len(groups) * 0.3))
    test_groups = set(groups[:n_test_groups])
    balanced['split'] = balanced['group_id'].apply(lambda g: 'test' if g in test_groups else 'train')
 
    print(f"  Train: {(balanced['split']=='train').sum()}")
    print(f"  Test: {(balanced['split']=='test').sum()}")
    print(f"  Train groups: {balanced[balanced['split']=='train']['group_id'].nunique()}")
    print(f"  Test groups ({n_test_groups}): {sorted(test_groups)}")
 
    # EXTRACT FEATURES from train set only
    print("\nExtracting features from train set...")
    train = balanced[balanced['split'] == 'train']
 
    features = pd.DataFrame([
        {**extract(np.load(rf"{BASE}\window_arrays\{r['window_id']}.npy"), cfg),
         **r[['window_id', 'label', 'group_id']]}
        for _, r in train.iterrows()
    ])
 
    features.to_csv(rf"{BASE}\features_organoid_17.csv", index=False)
 
    # SAVE TEST METADATA for later
    test = balanced[balanced['split'] == 'test']
    test[['window_id', 'file_id', 'group_id', 'label']].to_csv(rf"{BASE}\windows_metadata_test_organoid_17.csv", index=False)
 
    print(f"\n✓ {BASE}\\features_organoid_17.csv ({len(features)} samples, {len(features.columns)-3} features)")
    print(f"✓ {BASE}\\windows_metadata_test_organoid_17.csv ({len(test)} samples)")
 
if __name__ == "__main__":
    main()
