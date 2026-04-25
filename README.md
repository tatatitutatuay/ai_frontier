# AI Frontier ThaiSpoof Project

This repository contains a compact Thai voice anti-spoofing experiment for a biometrics mini-project.

## Structure

```text
data/raw/              # ignored raw audio datasets
ThaiSpoof/project/     # active runnable Python pipeline
ThaiSpoof/configs/     # editable experiment configs
ThaiSpoof/reports/     # final report and report template
ThaiSpoof/legacy/      # old reference scripts
ThaiSpoof/runs/        # ignored experiment outputs
tests/                 # unit tests
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ThaiSpoof/requirements.txt
```

## Common Commands

Check dataset detection:

```bash
python -m ThaiSpoof.project.run_experiment --stage summary
```

Run a tiny smoke experiment:

```bash
python -m ThaiSpoof.project.run_experiment --preset smoke --stage all
```

Run the Mac-sized baseline:

```bash
python -m ThaiSpoof.project.run_experiment --preset mac_small --stage all
```

Run the high-performance config:

```bash
python -m ThaiSpoof.project.run_experiment --config ThaiSpoof/configs/high_perf.json --stage all
```

## Final Report

The completed report is in `ThaiSpoof/reports/final_report.md`.
