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

The older scripts are left unchanged as reference material.

## Manual Step Required

Download ThaiSpoof from AI For Thai and place/extract it somewhere on your machine or Google Drive. I cannot complete that download automatically because it requires your AI For Thai account/session.

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

## Recommended Mac-Sized Run

Use this first on the M4 Air:

```bash
python -m ThaiSpoof.project.run_experiment \
  --data-root /path/to/ThaiSpoofData \
  --out-dir ThaiSpoof/runs/lfcc_small_mac \
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

## Recommended Colab Run

Use this when the dataset is in Google Drive:

```bash
python -m ThaiSpoof.project.run_experiment \
  --data-root /content/drive/MyDrive/ThaiSpoofData \
  --out-dir /content/drive/MyDrive/thaispoof_runs/lfcc_small_colab \
  --stage all \
  --feature lfcc \
  --model small_cnn \
  --train-genuine 3000 \
  --train-spoof 3000 \
  --test-genuine 1500 \
  --test-spoof 1500 \
  --dim-x 256 \
  --batch-size 16 \
  --epochs 30
```

After the baseline works, run the improvement:

```bash
python -m ThaiSpoof.project.run_experiment \
  --data-root /content/drive/MyDrive/ThaiSpoofData \
  --out-dir /content/drive/MyDrive/thaispoof_runs/lfcc_resnet_colab \
  --stage all \
  --feature lfcc \
  --model resnet_lite \
  --train-genuine 3000 \
  --train-spoof 3000 \
  --test-genuine 1500 \
  --test-spoof 1500 \
  --dim-x 256 \
  --batch-size 16 \
  --epochs 30
```

## Useful Stages

Create only the manifest:

```bash
python -m ThaiSpoof.project.run_experiment --data-root /path/to/ThaiSpoofData --stage split
```

Extract features only:

```bash
python -m ThaiSpoof.project.run_experiment --data-root /path/to/ThaiSpoofData --stage extract
```

Train from already extracted features:

```bash
python -m ThaiSpoof.project.run_experiment --data-root /path/to/ThaiSpoofData --stage train
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
- LFCC is the main feature because it matches the anti-spoofing literature better than plain MFCC for this project.
- `resnet_lite` is the improvement model. It is smaller than a full ResNet34 so it is safer on 16 GB RAM and Colab free runtimes.
- If your data folders are not detected, rename or wrap the folders so the path contains `genuine` or `spoof`.
