# Folder Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the ThaiSpoof project so code, datasets, reports, legacy scripts, and run artifacts are easier to scan.

**Architecture:** Keep the active Python package at `ThaiSpoof/project` so module imports remain stable. Move ignored raw audio into `data/raw`, move legacy reference scripts under `ThaiSpoof/legacy`, move report Markdown under `ThaiSpoof/reports`, and update configs/docs to use `data/raw` as the default data root.

**Tech Stack:** Git file moves, Python standard library tests, existing ThaiSpoof CLI.

---

### Task 1: Add Default Data Root Behavior

**Files:**
- Modify: `ThaiSpoof/project/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

Add a test asserting `resolve_experiment_config(preset_name="smoke")` defaults to `Path("data/raw")`.

- [ ] **Step 2: Run test**

Run: `.venv/bin/python -m unittest tests.test_config`

Expected: fail because the current default is `Path(".")`.

- [ ] **Step 3: Implement default data root**

Change `resolve_experiment_config` so it uses `values.setdefault("data_root", Path("data/raw"))`.

- [ ] **Step 4: Verify**

Run: `.venv/bin/python -m unittest tests.test_config`

Expected: pass.

### Task 2: Move Files And Data

**Files and directories:**
- Move ignored data: `genuine/` -> `data/raw/genuine/`
- Move ignored data: `Corpus-Spoof-VAJA/` -> `data/raw/Corpus-Spoof-VAJA/`
- Move tracked reports: `ThaiSpoof/final_report.md`, `ThaiSpoof/report_template.md` -> `ThaiSpoof/reports/`
- Move tracked legacy scripts: old top-level `ThaiSpoof/*.py` scripts and `ThaiSpoof/README.txt.txt` -> `ThaiSpoof/legacy/`

- [ ] **Step 1: Create destination folders**

Run: `mkdir -p data/raw ThaiSpoof/reports ThaiSpoof/legacy`

- [ ] **Step 2: Move ignored raw data**

Run: `mv genuine data/raw/genuine`

Run: `mv Corpus-Spoof-VAJA data/raw/Corpus-Spoof-VAJA`

- [ ] **Step 3: Move tracked report files**

Use `git mv` for `ThaiSpoof/final_report.md` and `ThaiSpoof/report_template.md`.

- [ ] **Step 4: Move tracked legacy scripts**

Use `git mv` for the old reference scripts and `ThaiSpoof/README.txt.txt`.

### Task 3: Update Paths And Documentation

**Files:**
- Modify: `.gitignore`
- Modify: `ThaiSpoof/PROJECT_README.md`
- Modify: `ThaiSpoof/configs/high_perf.json`
- Modify: `ThaiSpoof/reports/final_report.md`
- Create: `README.md`

- [ ] **Step 1: Update ignored paths**

Ignore `data/raw/` and keep `ThaiSpoof/runs/` ignored.

- [ ] **Step 2: Update docs and config paths**

Replace user-facing `--data-root .` examples with `--data-root data/raw`, update `high_perf.json`, and update report dataset folder references.

- [ ] **Step 3: Create root README**

Create a short root `README.md` that points to the active pipeline, final report, raw data location, and common commands.

### Task 4: Verify And Commit

- [ ] **Step 1: Run unit tests**

Run: `.venv/bin/python -m unittest discover -s tests`

Expected: all tests pass.

- [ ] **Step 2: Run dataset summary**

Run: `.venv/bin/python -m ThaiSpoof.project.run_experiment --stage summary`

Expected: 4,583 genuine and 4,583 spoof files detected from `data/raw`.

- [ ] **Step 3: Run smoke split**

Run: `.venv/bin/python -m ThaiSpoof.project.run_experiment --preset smoke --stage split --out-dir /tmp/thaispoof_restructure_split`

Expected: 8 train genuine, 8 train spoof, 4 test genuine, 4 test spoof.

- [ ] **Step 4: Commit**

Commit the restructure changes once verification passes.
