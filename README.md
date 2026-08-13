# Organoid Binary Classification Pipeline (No Event / Event)

This README explains how to run the organoid binary FSCV classification
pipeline end to end: from raw recordings + labels, to a trained RF+XGB+MLP
ensemble, to a held-out test evaluation, to interpretability figures.

17-feature pipeline (12 predecessor features + rise_time, decay_time, ox_red_ratio,
rise_slope, ox_red_lag) — same feature set as the iPSC binary pipeline.
No electrical stimulation in organoid data (forskolin raises spontaneous
event rate continuously, it isn't a discrete stimulus), so there's no
"stimulated" class here — binary only.

Classes: `0 = No Event (baseline)`, `1 = Event (spontaneous)`

---

## 1. Run order

| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 | `make_windows_organoid.py` | raw recordings (`.npy`/`.txt`/`.csv`) + labels CSV | `window_arrays/*.npy` + `windows_metadata.csv` |
| 2 | `extract_features_organoid.py` | `windows_metadata.csv` + `window_arrays/` | `features_organoid.csv` (train only) + `windows_metadata_test_organoid.csv` (held-out test) |
| 3 | `train_models_organoid.py` | `features_organoid.csv` + `window_arrays/` | `models_organoid/rf_model.pkl`, `xgb_model.pkl`, `mlp_model.pkl` (+ prints CV metrics) |
| 4 | `test_models_organoid.py` | `windows_metadata_test_organoid.csv` + `window_arrays/` + trained models | `results_organoid/{rf,xgb,mlp}_test.json` + `*_proba.npy` |
| 5 | `ensemble_organoid.py` | the three `*_proba.npy` files from step 4, all from `results_organoid/` | `results_organoid/ensemble_test.json` + `ensemble_proba.npy` |
| 6 | `analyse_organoid.py` | trained models (`models_organoid/`) + test set | interpretability figures (SHAP, permutation importance, gradient saliency) |

**Important workflow rule:** steps 1–3 are where you iterate. Steps 4–6 touch
the held-out test set and should only be run **once**, at the end, when the
CV results from step 3 look good.

`utils_organoid.py` is not run directly — it's a shared helper module
imported by steps 3–6 (metrics, data loading, `RANDOM_STATE`, `select_models`).

RF, XGB, and MLP are all trained together in one pass and read from the
same `models_organoid/` / `results_organoid/` folders throughout —
no separate or legacy model folder is used anywhere in this pipeline.

---

## 2. What needs changing before you run anything

Nothing here takes an input/output folder as a command-line argument — paths
are hardcoded at the top of each file.

### `make_windows_organoid.py`
```python
PLOT_DIR   = r"...\organoid data"   # folder of raw recording files
LABELS_CSV = r"...\organoid data output\FSCV_Labels.csv"   # your annotation CSV
BASE       = r"...\organoid data output"   # covers window_arrays/ and windows_metadata.csv
```

Two behaviours worth knowing about in this script:
- **Segments shorter than one window (2.0s)** are padded, centred on the
  labelled event, so they still produce a window instead of contributing
  nothing.
- **Baseline windows are sampled once per file**, from the merged region
  outside every event in that file combined — not once per individual
  labelled segment.

### `extract_features_organoid.py`, `utils_organoid.py`, `train_models_organoid.py`, `test_models_organoid.py`, `ensemble_organoid.py`, `analyse_organoid.py`
Each has a `BASE` constant near the top:
```python
BASE = r"C:\Users\julie\OneDrive - Imperial College London\organoid data output"
```
**This must be identical across all seven files (including `make_windows_organoid.py`)** — each script writes into `BASE\...` and the next one reads from `BASE\...`.

### `fscv_config_organoid.yaml`
Check these values match your recording setup before running step 1:
```yaml
fscv_hz: 10.0                 # sampling rate
stride: 5                     # step size in frames between windows
max_nothing: 50               # max baseline windows extracted per file
v_oxidation_start: 150        # oxidation band row indices (organoid-specific — peak confirmed at row 213, within this band)
v_oxidation_end: 300
v_reduction_start: 800        # reduction band row indices
v_reduction_end: 1000
balance_ratio: 2              # baseline:event ratio when balancing classes
```

---

## 3. How to run

From the folder containing all the scripts and `fscv_config_organoid.yaml`:

```bash
python make_windows_organoid.py --config fscv_config_organoid.yaml
python extract_features_organoid.py --config fscv_config_organoid.yaml
python train_models_organoid.py all
python test_models_organoid.py all
python ensemble_organoid.py
python analyse_organoid.py
```

`train_models_organoid.py` and `test_models_organoid.py` accept `rf`,
`xgb`, `mlp`, or `all` — run `all` for the full pipeline, or a single model
name to re-run just one. With no arguments, either script drops into an
interactive prompt asking which model(s) to run (options limited to RF/XGB/MLP
— CNN and LSTM aren't part of this pipeline).

---

## 4. Outputs you'll end up with (inside `BASE`)

```
organoid data output/
├── window_arrays/                            (step 1)
├── windows_metadata.csv                      (step 1 — all windows)
├── features_organoid.csv                  (step 2 — train features, 17 cols + label/group/window_id)
├── windows_metadata_test_organoid.csv     (step 2 — held-out test windows)
├── models_organoid/
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── mlp_model.pkl
│   └── mlp_oof_ytrue.npy / mlp_oof_yproba.npy
└── results_organoid/
    ├── rf_test.json / rf_proba.npy
    ├── xgb_test.json / xgb_proba.npy
    ├── mlp_test.json / mlp_proba.npy
    └── ensemble_test.json / ensemble_proba.npy   (step 5 — final result)
```

`ensemble_test.json` is the final number to report for this bundle
(soft-voting RF+XGB+MLP, F1_macro on held-out test).

---

## 5. Quick sanity checks

- Step 1 print-out should show a plausible split of baseline vs. event
  files and a non-zero window count for both classes. Also check the
  "Segments padded to minimum window length" count printed at the end —
  a very high number relative to your total event count is worth a second
  look at your label durations.
- Step 2 print-out shows the group-aware train/test split — check `Test
  groups` isn't empty and doesn't overlap with train groups.
- Step 3 prints CV `F1_macro` per model — this is your working number,
  iterate here.
- Steps 4–6 should only be run once real CV results look acceptable.
- Step 5 requires all three `*_proba.npy` files (`rf`, `xgb`, `mlp`) to
  exist in `results_organoid/` first — all from the same folder.
