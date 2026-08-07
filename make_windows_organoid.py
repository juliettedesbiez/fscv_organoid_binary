"""
FSCV Window Generator — Organoid (BINARY)
Usage: python make_windows_organoid.py [--config fscv_config_organoid.yaml]

Classes: 0=baseline, 1=spontaneous
(Labelled via Labelling_App.py, which is 3-class capable but only
0/1 are ever assigned for organoid data — label 2 is filtered out defensively below.)
"""

import os, argparse
import numpy as np
import pandas as pd
import yaml

# Paths - organoid-specific
PLOT_DIR   = r"C:\Users\julie\OneDrive - Imperial College London\organoid data"       # <-- confirm exact folder name
LABELS_CSV = r"C:\Users\julie\OneDrive - Imperial College London\organoid data output\FSCV_Labels_July.csv"  # <-- confirm exact filename from Labelling_App_organoid.py

# BASE covers everything this script writes — window_arrays/ and windows_metadata.csv both live under it
BASE       = r"C:\Users\julie\OneDrive - Imperial College London\organoid data output"
WINDOW_DIR = rf"{BASE}\window_arrays"

def load_config(path="fscv_config_organoid.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_arr(path):
    arr = np.load(path) if path.endswith('.npy') else np.loadtxt(path)
    return arr[np.newaxis, :] if arr.ndim == 1 else arr

def process_file(arr, fname, file_id, group_id, file_labels, meta_rows,
                 w_counter, label_val, cfg, window_frames, stride, max_nothing):
    """Extract labelled windows from one file."""
    nT = arr.shape[1]
    for _, row in file_labels.iterrows():
        f_start = int(row['start_time'] * cfg['fscv_hz'])
        f_end   = int(row['end_time']   * cfg['fscv_hz'])

        # Spontaneous windows
        for f0 in range(f_start, max(f_start, f_end - window_frames + 1), stride):
            window = arr[:, f0:f0+window_frames]
            if window.shape[1] != window_frames: continue
            w_counter[fname] = w_counter.get(fname, 0) + 1
            wid = f"{file_id}_w{w_counter[fname]:04d}"
            np.save(f"{WINDOW_DIR}/{wid}.npy", window)
            meta_rows.append({'window_id': wid, 'file_id': file_id,
                              'group_id': group_id, 'label': label_val,
                              'start_frame': int(f0), 'end_frame': int(f0+window_frames)})

        # Baseline windows from the same file (outside the signal region)
        for fs_int, fe_int in [(0, f_start), (f_end, nT)]:
            positions = list(range(fs_int, fe_int - window_frames + 1, stride))
            for f0 in np.linspace(0, len(positions)-1, min(len(positions), max_nothing), dtype=int):
                f0 = positions[int(f0)]
                window = arr[:, f0:f0+window_frames]
                if window.shape[1] != window_frames: continue
                w_counter[fname] = w_counter.get(fname, 0) + 1
                wid = f"{file_id}_w{w_counter[fname]:04d}"
                np.save(f"{WINDOW_DIR}/{wid}.npy", window)
                meta_rows.append({'window_id': wid, 'file_id': file_id,
                                  'group_id': group_id, 'label': 0,
                                  'start_frame': int(f0), 'end_frame': int(f0+window_frames)})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='fscv_config_organoid.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    fscv_hz       = cfg['fscv_hz']
    stride        = cfg['stride']
    max_nothing   = cfg['max_nothing']
    window_frames = int(2.0 * fscv_hz)   # 2s window, same as iPSC -> 1100x20
    bg_frames     = int(5.0 * fscv_hz)

    print("=" * 70)
    print("FSCV WINDOW GENERATOR (ORGANOID BINARY)")
    print(f"Config: {args.config} | Hz={fscv_hz} | Window={window_frames}f (2.0s) | Stride={stride} | BG={bg_frames}f")
    print("=" * 70)

    os.makedirs(WINDOW_DIR, exist_ok=True)

    meta = pd.read_csv(LABELS_CSV)
    meta = meta[meta['label'].isin([0, 1])].copy()   # defensive filter — drops any stray label=2
    meta['label'] = meta['label'].astype(int)

    baseline_files  = set(meta[meta['label'] == 0]['plot_file'].unique())
    spont_files     = set(meta[meta['label'] == 1]['plot_file'].unique())
    available       = {f for f in os.listdir(PLOT_DIR) if f.lower().endswith(('.npy', '.txt', '.csv'))}

    print(f"\nFiles: {len(baseline_files)} baseline-only, {len(spont_files)} spontaneous")

    meta_rows, w_counter = [], {}

    # Process pure baseline files
    print("\nProcessing baseline files...")
    for fname in sorted(baseline_files & available):
        try:
            arr = load_arr(os.path.join(PLOT_DIR, fname))
            arr = arr - arr[:, :bg_frames].mean(axis=1, keepdims=True)
            file_id  = fname[:-4]
            group_id = '_'.join(file_id.split('_')[:2])   # e.g. "231211_organoid4"
            nT = arr.shape[1]
            positions = list(range(0, nT - window_frames + 1, stride))
            for f0 in np.linspace(0, len(positions)-1, min(len(positions), max_nothing), dtype=int):
                f0 = positions[int(f0)]
                window = arr[:, f0:f0+window_frames]
                if window.shape[1] != window_frames: continue
                w_counter[fname] = w_counter.get(fname, 0) + 1
                wid = f"{file_id}_w{w_counter[fname]:04d}"
                np.save(f"{WINDOW_DIR}/{wid}.npy", window)
                meta_rows.append({'window_id': wid, 'file_id': file_id,
                                  'group_id': group_id, 'label': 0,
                                  'start_frame': int(f0), 'end_frame': int(f0+window_frames)})
        except Exception as e:
            print(f"  Skipped {fname}: {e}")

    # Process spontaneous files (label=1)
    print("Processing spontaneous files...")
    for fname in sorted(spont_files & available):
        try:
            arr = load_arr(os.path.join(PLOT_DIR, fname))
            arr = arr - arr[:, :bg_frames].mean(axis=1, keepdims=True)
            file_id  = fname[:-4]
            group_id = '_'.join(file_id.split('_')[:2])
            file_labels = meta[(meta['plot_file'] == fname) & (meta['label'] == 1)]
            process_file(arr, fname, file_id, group_id, file_labels, meta_rows,
                         w_counter, 1, cfg, window_frames, stride, max_nothing)
        except Exception as e:
            print(f"  Skipped {fname}: {e}")

    df = pd.DataFrame(meta_rows)
    for col in ['window_id', 'file_id', 'group_id']: df[col] = df[col].astype(str)
    df['label'] = df['label'].astype(int)
    df.to_csv(rf"{BASE}\windows_metadata.csv", index=False)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"Total windows: {len(df)}")
    print(f"  Baseline (0):    {(df['label']==0).sum()}")
    print(f"  Spontaneous (1): {(df['label']==1).sum()}")
    print(f"\nSaved: windows_metadata.csv, {WINDOW_DIR}/ ({len(df)} files)")

if __name__ == "__main__":
    main()
