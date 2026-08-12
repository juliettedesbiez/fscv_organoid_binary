"""
FSCV Window Generator — Organoid (BINARY)
Usage: python make_windows_organoid.py [--config fscv_config_organoid.yaml]

Classes: 0=baseline, 1=spontaneous
(Labelled via Labelling_App.py, which is 3-class capable but only
0/1 are ever assigned for organoid data — label 2 is filtered out defensively below.)

Two fixes applied post-relabelling:
1. Segments shorter than one window (2.0s) are padded, centred on the
   labelled event. Post relabelling, most spontaneous segments came in
   under 2.0s (median ~1.25s), which meant ~79% were producing zero
   windows under the old logic.
2. Baseline windows are now sampled ONCE per file, from the merged region
   outside every segment combined -- not once per individual segment. The
   old logic re-sampled up to max_nothing baseline windows on every pass
   through the per-segment loop, so a file with (for example) 9 separate
   spontaneous segments could silently produce up to 9x the intended
   baseline windows. This was invisible with the old labels (mostly 1-2
   segments per file) but badly inflated baseline counts once relabelling
   produced files with many more segments each.
"""

import os, argparse
import numpy as np
import pandas as pd
import yaml

# Paths - organoid-specific
PLOT_DIR   = r"C:\Users\julie\OneDrive - Imperial College London\organoid data"       # <-- confirm exact folder name
LABELS_CSV = r"C:\Users\julie\OneDrive - Imperial College London\organoid data output\FSCV_Labels_July.csv"  # <-- confirm exact filename from Labelling_App_organoid.py

# BASE covers everything this script writes — window_arrays/ and windows_metadata.csv both live under it
BASE       = r"C:\Users\julie\OneDrive - Imperial College London\organoid data output retrain 3"
WINDOW_DIR = rf"{BASE}\window_arrays"

def load_config(path="fscv_config_organoid.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_arr(path):
    arr = np.load(path) if path.endswith('.npy') else np.loadtxt(path)
    arr = arr[np.newaxis, :] if arr.ndim == 1 else arr
    return -arr

def process_file(arr, fname, file_id, group_id, file_labels, meta_rows,
                 w_counter, label_val, cfg, window_frames, stride, max_nothing,
                 pad_counter=None):
    """Extract labelled windows from one file."""
    nT = arr.shape[1]
    padded_spans = []   # collect (f_start, f_end) after padding, for merged baseline exclusion

    # --- Signal windows: still generated per-segment, as before ---
    for _, row in file_labels.iterrows():
        f_start = int(row['start_time'] * cfg['fscv_hz'])
        f_end   = int(row['end_time']   * cfg['fscv_hz'])

        # Pad segments shorter than one window so they still produce a
        # window, centred on the labelled event. Clipped to file boundaries
        # for events near the very start or end of a recording.
        if f_end - f_start < window_frames:
            pad = window_frames - (f_end - f_start)
            pad_before = pad // 2
            f_start = max(0, f_start - pad_before)
            f_end = f_start + window_frames
            if f_end > nT:
                f_end = nT
                f_start = max(0, f_end - window_frames)
            if pad_counter is not None:
                pad_counter[0] += 1

        padded_spans.append((f_start, f_end))

        for f0 in range(f_start, max(f_start, f_end - window_frames + 1), stride):
            window = arr[:, f0:f0+window_frames]
            if window.shape[1] != window_frames: continue
            w_counter[fname] = w_counter.get(fname, 0) + 1
            wid = f"{file_id}_w{w_counter[fname]:04d}"
            np.save(f"{WINDOW_DIR}/{wid}.npy", window)
            meta_rows.append({'window_id': wid, 'file_id': file_id,
                              'group_id': group_id, 'label': label_val,
                              'start_frame': int(f0), 'end_frame': int(f0+window_frames)})

    # --- Baseline windows: sampled ONCE per file, from the merged region
    # outside every (padded) segment combined ---
    if not padded_spans:
        baseline_candidates = list(range(0, nT - window_frames + 1, stride))
    else:
        padded_spans.sort()
        merged = []
        for s, e in padded_spans:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        baseline_candidates = []
        prev_end = 0
        for s, e in merged:
            baseline_candidates += list(range(prev_end, max(prev_end, s - window_frames + 1), stride))
            prev_end = e
        baseline_candidates += list(range(prev_end, max(prev_end, nT - window_frames + 1), stride))

    n_take = min(len(baseline_candidates), max_nothing)
    for idx in np.linspace(0, len(baseline_candidates) - 1, n_take, dtype=int) if baseline_candidates else []:
        f0 = baseline_candidates[int(idx)]
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

    meta_rows, w_counter, pad_counter = [], {}, [0]

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
                         w_counter, 1, cfg, window_frames, stride, max_nothing, pad_counter)
        except Exception as e:
            print(f"  Skipped {fname}: {e}")

    df = pd.DataFrame(meta_rows)
    for col in ['window_id', 'file_id', 'group_id']: df[col] = df[col].astype(str)
    df['label'] = df['label'].astype(int)
    df.to_csv(rf"{BASE}\windows_metadata.csv", index=False)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"Segments padded to minimum window length: {pad_counter[0]}")
    print(f"Total windows: {len(df)}")
    print(f"  Baseline (0):    {(df['label']==0).sum()}")
    print(f"  Spontaneous (1): {(df['label']==1).sum()}")
    print(f"\nSaved: windows_metadata.csv, {WINDOW_DIR}/ ({len(df)} files)")

if __name__ == "__main__":
    main()
