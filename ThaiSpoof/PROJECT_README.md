# ThaiSpoof Voice Anti-Spoofing Project

This project builds a practical fake-speech detector for the Biometrics mini-project. It focuses on ThaiSpoof only and is designed to run on a MacBook Air M4 with 16 GB RAM or on Google Colab.

## What Was Added

New runnable pipeline:

- `ThaiSpoof/project/config.py` - experiment configuration
- `ThaiSpoof/project/dataset.py` - audio discovery, class inference, balanced splits, manifest writing
- `ThaiSpoof/project/features.py` - LFCC/MFCC extraction without the old hard-coded paths
- `ThaiSpoof/project/models.py` - `small_cnn` baseline and `resnet_lite` improvement model
- `ThaiSpoof/project/train.py` - training, validation, testing, metrics export
- `ThaiSpoof/project/run_experiment.py` - command-line entry point
- `ThaiSpoof/configs/high_perf.json` - editable high-performance run config

The older scripts are left unchanged as reference material.

## Dataset Layout

Download ThaiSpoof from AI For Thai and place/extract it somewhere on your machine or Google Drive. I cannot complete that download automatically because it requires your AI For Thai account/session.

This workspace currently supports the downloaded local layout directly from the project root:

```text
ai_frontier/
  genuine/
    G1/
      *.wav
  Corpus-Spoof-VAJA/
    Train/
      *.wav
    Test/
      *.wav
```

The pipeline searches recursively and detects classes from folder names. These names work:

- Genuine: `genuine`, `real`, `bona_fide`, `bonafide`
- Spoof: `spoof`, `spoofed`, `synthetic`, `fake`, `tts_vaja_spoof`, `mms_spoof`, `pitch shift spoofed audio`, `speed change`, etc.

Example accepted structure:

```text
ThaiSpoofData/
  genuine/
    *.wav
  tts_vaja_spoof/
    *.wav
  mms_spoof/
    *.wav
  pitch_shift_spoof/
    *.wav
```

The exact structure can be nested; the important part is that at least one path segment identifies genuine or spoof.

## Install

Mac or local Python:

```bash
cd /Users/tatatitutatuay/tatatitutatuay/CEDT/ai_frontier
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ThaiSpoof/requirements.txt
```

Google Colab:

```bash
!pip install soundfile scipy
```

Colab usually already includes TensorFlow, NumPy, and SciPy. If TensorFlow import fails, run:

```bash
!pip install tensorflow
```

## Quick Check

Inspect the downloaded dataset before feature extraction or training:

```bash
python3 -m ThaiSpoof.project.run_experiment --data-root . --stage summary
```

This should report nonzero counts for both `genuine` and `spoof`.

## Presets And Config Files

The runner resolves settings in this order:

1. Built-in defaults.
2. `--preset`.
3. JSON `--config`.
4. Explicit command-line flags.

Available presets:

| Preset | Use case |
| --- | --- |
| `smoke` | Tiny wiring check on the Mac before a real run |
| `mac_small` | Safe MacBook Air M4 baseline |
| `balanced_medium` | Larger local CPU/RAM run |
| `high_perf` | Strong workstation or GPU runtime |

Run the tiny smoke path first:

```bash
python3 -m ThaiSpoof.project.run_experiment \
  --data-root . \
  --preset smoke \
  --stage all
```

## Recommended Mac-Sized Run

Use this first on the M4 Air:

```bash
python3 -m ThaiSpoof.project.run_experiment \
  --data-root . \
  --preset mac_small \
  --stage all
```

The expanded equivalent is:

```bash
python3 -m ThaiSpoof.project.run_experiment \
  --data-root . \
  --preset mac_small \
  --stage all \
  --feature lfcc \
  --model small_cnn \
  --train-genuine 1000 \
  --train-spoof 1000 \
  --test-genuine 500 \
  --test-spoof 500 \
  --dim-x 192 \
  --batch-size 8 \
  --epochs 15
```

## High-Performance Run

Edit `ThaiSpoof/configs/high_perf.json` when running on a stronger computer. The included config starts with:

```json
{
  "preset": "high_perf",
  "data_root": ".",
  "out_dir": "ThaiSpoof/runs/lfcc_high_perf",
  "feature": "lfcc",
  "model": "resnet_lite",
  "train_genuine": 3000,
  "train_spoof": 3000,
  "test_genuine": 1500,
  "test_spoof": 1500,
  "dim_x": 256,
  "batch_size": 32,
  "epochs": 40,
  "patience": 8
}
```

Run it with:

```bash
python3 -m ThaiSpoof.project.run_experiment \
  --config ThaiSpoof/configs/high_perf.json \
  --stage all
```

## Useful Stages

Summarize detected audio:

```bash
python3 -m ThaiSpoof.project.run_experiment --data-root . --stage summary
```

Create only the manifest:

```bash
python3 -m ThaiSpoof.project.run_experiment --data-root . --preset smoke --stage split
```

Extract features only:

```bash
python3 -m ThaiSpoof.project.run_experiment --data-root . --preset smoke --stage extract
```

Train from already extracted features:

```bash
python3 -m ThaiSpoof.project.run_experiment --data-root . --preset smoke --stage train
```

Force a manifest or feature-cache rebuild:

```bash
python3 -m ThaiSpoof.project.run_experiment --data-root . --preset smoke --stage all --force-split --force-extract
```

## Outputs

Each run writes:

```text
ThaiSpoof/runs/<run_name>/
  manifest.csv
  features/<feature>/
    LFCC_Train_genuine_*.pkl
    LFCC_Train_spoof_*.pkl
    LFCC_Test_genuine_*.pkl
    LFCC_Test_spoof_*.pkl
    SUMMARY_counts.csv
  models/
    lfcc_small_cnn.keras
  results/
    lfcc_small_cnn_history.csv
    lfcc_small_cnn_metrics.csv
    lfcc_small_cnn_summary.md
```

The metrics CSV contains accuracy, balanced accuracy, precision, recall, F1, EER, and confusion counts for train, validation, and test.

## Experiment Table

Use the final metrics to fill this table in the report:

| Experiment | Feature | Model | Accuracy | F1-score | EER |
| --- | --- | --- | --- | --- | --- |
| Baseline | LFCC | small_cnn | | | |
| Improvement | LFCC | resnet_lite | | | |
| Optional | MFCC | small_cnn | | | |

## Notes

- Start with the Mac-sized run. It is intentionally small so the full pipeline can be debugged quickly.
- Feature extraction is cached by output directory, feature type, and split counts. Re-running the same config reuses matching pickle files unless `--force-extract` is set.
- LFCC is the main feature because it matches the anti-spoofing literature better than plain MFCC for this project.
- `resnet_lite` is the improvement model. It is smaller than a full ResNet34 so it is safer on 16 GB RAM and Colab free runtimes.
- If your data folders are not detected, rename or wrap the folders so the path contains `genuine` or `spoof`.
