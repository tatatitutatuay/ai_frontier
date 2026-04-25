# ThaiSpoof Config Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Mac-safe and high-performance experiment configuration paths for the ThaiSpoof pipeline.

**Architecture:** Keep the existing `ThaiSpoof/project` package and extend it with deterministic config resolution, dataset summary support, and feature-cache checks. Built-in presets provide quick hardware profiles, JSON config files support stronger machines, and command-line overrides remain the final source of truth.

**Tech Stack:** Python standard library, `unittest`, existing NumPy/SciPy/SoundFile/TensorFlow pipeline.

---

## File Structure

- Modify `ThaiSpoof/project/config.py`: add built-in presets, JSON config loading, override filtering, and `resolve_experiment_config`.
- Modify `ThaiSpoof/project/features.py`: add feature-cache detection and optional skip behavior.
- Modify `ThaiSpoof/project/run_experiment.py`: add `--preset`, `--config`, `summary` stage, `--force-split`, and `--force-extract`; route argument resolution through `config.py`.
- Modify `ThaiSpoof/PROJECT_README.md`: document current local dataset layout, smoke run, Mac run, and high-performance JSON config.
- Modify `tests/test_config.py`: cover preset resolution, JSON config loading, and CLI-style override precedence.
- Modify `tests/test_dataset.py`: cover current `genuine/` plus `Corpus-Spoof-VAJA/` layout detection with tiny temp files.
- Create `tests/test_features.py`: cover cache detection without reading audio.

### Task 1: Config Presets And JSON Loading

**Files:**
- Modify: `ThaiSpoof/project/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for preset and JSON config behavior**

Add these tests to `tests/test_config.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path

from ThaiSpoof.project.config import ExperimentConfig, load_config_file, resolve_experiment_config


class ExperimentConfigTest(unittest.TestCase):
    def test_normalizes_feature_and_model_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ExperimentConfig(
                data_root=Path(tmp) / "data",
                out_dir=Path(tmp) / "out",
                feature="LFCC",
                model="Small_CNN",
            )

        self.assertEqual(cfg.feature, "lfcc")
        self.assertEqual(cfg.model, "small_cnn")

    def test_rejects_non_positive_subset_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                ExperimentConfig(
                    data_root=Path(tmp) / "data",
                    out_dir=Path(tmp) / "out",
                    train_genuine=0,
                )

    def test_feature_dir_uses_feature_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ExperimentConfig(
                data_root=Path(tmp) / "data",
                out_dir=Path(tmp) / "out",
                feature="mfcc",
            )

        self.assertEqual(cfg.feature_dir, Path(tmp) / "out" / "features" / "mfcc")

    def test_smoke_preset_uses_tiny_mac_safe_counts(self):
        cfg = resolve_experiment_config(
            preset_name="smoke",
            config_file=None,
            overrides={"data_root": Path(".")},
        )

        self.assertEqual(cfg.train_genuine, 8)
        self.assertEqual(cfg.train_spoof, 8)
        self.assertEqual(cfg.test_genuine, 4)
        self.assertEqual(cfg.test_spoof, 4)
        self.assertEqual(cfg.batch_size, 4)
        self.assertEqual(cfg.epochs, 2)

    def test_json_config_overrides_preset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "high_perf.json"
            path.write_text(
                json.dumps(
                    {
                        "preset": "mac_small",
                        "data_root": ".",
                        "out_dir": "ThaiSpoof/runs/custom",
                        "model": "resnet_lite",
                        "batch_size": 32,
                        "epochs": 40,
                    }
                ),
                encoding="utf-8",
            )

            cfg = resolve_experiment_config(
                preset_name=None,
                config_file=path,
                overrides={},
            )

        self.assertEqual(cfg.model, "resnet_lite")
        self.assertEqual(cfg.batch_size, 32)
        self.assertEqual(cfg.epochs, 40)
        self.assertEqual(cfg.out_dir, Path("ThaiSpoof/runs/custom"))

    def test_cli_overrides_json_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps({"preset": "smoke", "data_root": ".", "batch_size": 4}),
                encoding="utf-8",
            )

            cfg = resolve_experiment_config(
                preset_name=None,
                config_file=path,
                overrides={"batch_size": 12, "epochs": 5},
            )

        self.assertEqual(cfg.batch_size, 12)
        self.assertEqual(cfg.epochs, 5)

    def test_load_config_file_rejects_unknown_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"data_root": ".", "unknown": 1}), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_config_file(path)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run config tests and verify failure**

Run: `python3 -m unittest tests.test_config`

Expected: fail because `load_config_file` and `resolve_experiment_config` do not exist yet.

- [ ] **Step 3: Implement config presets and resolution**

In `ThaiSpoof/project/config.py`, add:

```python
import json
from typing import Any, Mapping

CONFIG_FIELDS = set(ExperimentConfig.__dataclass_fields__)
PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "out_dir": Path("ThaiSpoof/runs/smoke_lfcc_small"),
        "train_genuine": 8,
        "train_spoof": 8,
        "test_genuine": 4,
        "test_spoof": 4,
        "dim_x": 96,
        "batch_size": 4,
        "epochs": 2,
        "patience": 2,
    },
    "mac_small": {
        "out_dir": Path("ThaiSpoof/runs/lfcc_small_mac"),
        "train_genuine": 1000,
        "train_spoof": 1000,
        "test_genuine": 500,
        "test_spoof": 500,
        "dim_x": 192,
        "batch_size": 8,
        "epochs": 15,
        "patience": 4,
    },
    "balanced_medium": {
        "out_dir": Path("ThaiSpoof/runs/lfcc_medium"),
        "train_genuine": 2000,
        "train_spoof": 2000,
        "test_genuine": 1000,
        "test_spoof": 1000,
        "dim_x": 224,
        "batch_size": 16,
        "epochs": 25,
        "patience": 6,
    },
    "high_perf": {
        "out_dir": Path("ThaiSpoof/runs/lfcc_high_perf"),
        "model": "resnet_lite",
        "train_genuine": 3000,
        "train_spoof": 3000,
        "test_genuine": 1500,
        "test_spoof": 1500,
        "dim_x": 256,
        "batch_size": 32,
        "epochs": 40,
        "patience": 8,
    },
}


def _validate_keys(values: Mapping[str, Any], source: str) -> None:
    allowed = CONFIG_FIELDS | {"preset"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown config key(s) in {source}: {', '.join(unknown)}")


def load_config_file(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ValueError("config file must contain a JSON object")
    _validate_keys(values, str(path))
    return values


def preset_values(name: str | None) -> dict[str, Any]:
    if not name:
        return {}
    key = name.lower()
    if key not in PRESETS:
        raise ValueError(f"unknown preset '{name}'. Choose one of: {', '.join(sorted(PRESETS))}")
    return dict(PRESETS[key])


def _clean_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    if not overrides:
        return {}
    _validate_keys(overrides, "overrides")
    return {key: value for key, value in overrides.items() if value is not None}


def resolve_experiment_config(
    preset_name: str | None = None,
    config_file: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    values: dict[str, Any] = {}
    values.update(preset_values(preset_name))

    if config_file:
        file_values = load_config_file(config_file)
        file_preset = file_values.pop("preset", None)
        if file_preset:
            values.update(preset_values(str(file_preset)))
        values.update(file_values)

    values.update(_clean_overrides(overrides))
    values.setdefault("data_root", Path("."))
    return ExperimentConfig(**values)
```

- [ ] **Step 4: Run config tests and verify pass**

Run: `python3 -m unittest tests.test_config`

Expected: all tests in `tests.test_config` pass.

### Task 2: Dataset Summary And CLI Config Resolution

**Files:**
- Modify: `ThaiSpoof/project/run_experiment.py`
- Modify: `tests/test_dataset.py`

- [ ] **Step 1: Write failing dataset layout test**

Add this test to `tests/test_dataset.py`:

```python
    def test_collect_audio_detects_current_workspace_layout_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            touch(root / "genuine" / "G1" / "thai0001.wav")
            touch(root / "Corpus-Spoof-VAJA" / "Train" / "thai_000001.wav")

            items = collect_audio(root)

        labels = sorted(item.label for item in items)
        attacks = sorted(item.attack_type for item in items)
        self.assertEqual(labels, ["genuine", "spoof"])
        self.assertIn("corpus_spoof_vaja", attacks)
```

- [ ] **Step 2: Run dataset tests and verify failure if layout alias is missing**

Run: `python3 -m unittest tests.test_dataset`

Expected: fail because `Corpus-Spoof-VAJA` is not yet explicitly listed as a spoof alias if substring detection does not catch it.

- [ ] **Step 3: Implement CLI config resolution and summary stage**

In `ThaiSpoof/project/run_experiment.py`:

- Import `PRESETS` and `resolve_experiment_config`.
- Add `summary` to `--stage`.
- Add `--preset` with choices from `PRESETS`.
- Add `--config`.
- Add `--force-split`.
- Add `--force-extract`.
- Set configurable parser defaults to `None` so CLI overrides are distinguishable.
- Add `build_config_overrides(args)` returning only config fields.
- Add `log_dataset_summary(config)` that calls `collect_audio`, counts labels and attack types, and logs the totals.

- [ ] **Step 4: Run config and dataset tests**

Run: `python3 -m unittest tests.test_config tests.test_dataset`

Expected: both modules pass.

### Task 3: Feature Cache Reuse

**Files:**
- Modify: `ThaiSpoof/project/features.py`
- Modify: `ThaiSpoof/project/run_experiment.py`
- Create: `tests/test_features.py`

- [ ] **Step 1: Write failing feature-cache tests**

Create `tests/test_features.py`:

```python
import tempfile
import unittest
from pathlib import Path

from ThaiSpoof.project.features import feature_groups_exist


class FeatureCacheTest(unittest.TestCase):
    def test_feature_groups_exist_when_all_expected_pickles_are_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in [
                "LFCC_Train_genuine_8.pkl",
                "LFCC_Train_spoof_8.pkl",
                "LFCC_Test_genuine_4.pkl",
                "LFCC_Test_spoof_4.pkl",
            ]:
                (root / name).write_bytes(b"cache")

            self.assertTrue(feature_groups_exist(root, "lfcc", 8, 8, 4, 4))

    def test_feature_groups_missing_when_count_does_not_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "LFCC_Train_genuine_7.pkl").write_bytes(b"cache")

            self.assertFalse(feature_groups_exist(root, "lfcc", 8, 8, 4, 4))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run feature tests and verify failure**

Run: `python3 -m unittest tests.test_features`

Expected: fail because `feature_groups_exist` does not exist yet.

- [ ] **Step 3: Implement feature-cache detection**

In `ThaiSpoof/project/features.py`, add:

```python
def feature_groups_exist(
    out_dir: Path,
    feature: str,
    train_genuine: int,
    train_spoof: int,
    test_genuine: int,
    test_spoof: int,
) -> bool:
    out_dir = Path(out_dir)
    prefix = feature.upper()
    expected = [
        out_dir / f"{prefix}_Train_genuine_{train_genuine}.pkl",
        out_dir / f"{prefix}_Train_spoof_{train_spoof}.pkl",
        out_dir / f"{prefix}_Test_genuine_{test_genuine}.pkl",
        out_dir / f"{prefix}_Test_spoof_{test_spoof}.pkl",
    ]
    return all(path.exists() for path in expected)
```

In `ThaiSpoof/project/run_experiment.py`, before `save_feature_groups`, call `feature_groups_exist(...)`; if it returns true and `--force-extract` is false, log cache reuse and skip extraction.

- [ ] **Step 4: Run feature tests and verify pass**

Run: `python3 -m unittest tests.test_features`

Expected: feature-cache tests pass.

### Task 4: Documentation And Verification

**Files:**
- Modify: `ThaiSpoof/PROJECT_README.md`

- [ ] **Step 1: Update docs with concrete commands**

Add examples:

```bash
python3 -m ThaiSpoof.project.run_experiment --data-root . --stage summary
python3 -m ThaiSpoof.project.run_experiment --data-root . --preset smoke --stage all
python3 -m ThaiSpoof.project.run_experiment --data-root . --preset mac_small --stage all
python3 -m ThaiSpoof.project.run_experiment --config ThaiSpoof/configs/high_perf.json --stage all
```

Add an example JSON block for high-performance runs.

- [ ] **Step 2: Run full unit tests**

Run: `python3 -m unittest discover -s tests`

Expected: all unit tests pass.

- [ ] **Step 3: Run dataset summary smoke command**

Run: `python3 -m ThaiSpoof.project.run_experiment --data-root . --stage summary`

Expected: logs report nonzero genuine and spoof counts.

- [ ] **Step 4: Run a tiny split command**

Run: `python3 -m ThaiSpoof.project.run_experiment --data-root . --preset smoke --stage split --out-dir /tmp/thaispoof_smoke_split`

Expected: manifest is written with 8 train genuine, 8 train spoof, 4 test genuine, and 4 test spoof.
