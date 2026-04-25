from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SUPPORTED_FEATURES = {"lfcc", "mfcc"}
SUPPORTED_MODELS = {"small_cnn", "resnet_lite"}


@dataclass(frozen=True)
class ExperimentConfig:
    data_root: Path
    out_dir: Path = Path("ThaiSpoof/runs/thaispoof_lfcc")
    feature: str = "lfcc"
    model: str = "small_cnn"
    train_genuine: int = 3000
    test_genuine: int = 1500
    train_spoof: int = 3000
    test_spoof: int = 1500
    seed: int = 42
    target_sr: int = 16000
    dim_x: int = 256
    dim_y: int = 60
    batch_size: int = 16
    epochs: int = 30
    validation_fraction: float = 0.2
    learning_rate: float = 1e-4
    patience: int = 6

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root).expanduser())
        object.__setattr__(self, "out_dir", Path(self.out_dir).expanduser())
        object.__setattr__(self, "feature", self.feature.lower())
        object.__setattr__(self, "model", self.model.lower())

        if self.feature not in SUPPORTED_FEATURES:
            raise ValueError(f"feature must be one of {sorted(SUPPORTED_FEATURES)}")
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(f"model must be one of {sorted(SUPPORTED_MODELS)}")

        positive_ints = {
            "train_genuine": self.train_genuine,
            "test_genuine": self.test_genuine,
            "train_spoof": self.train_spoof,
            "test_spoof": self.test_spoof,
            "target_sr": self.target_sr,
            "dim_x": self.dim_x,
            "dim_y": self.dim_y,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
        }
        bad = [name for name, value in positive_ints.items() if value <= 0]
        if bad:
            raise ValueError(f"positive integer required for: {', '.join(bad)}")

        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between 0 and 0.5")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    @property
    def manifest_path(self) -> Path:
        return self.out_dir / "manifest.csv"

    @property
    def feature_dir(self) -> Path:
        return self.out_dir / "features" / self.feature

    @property
    def model_dir(self) -> Path:
        return self.out_dir / "models"

    @property
    def results_dir(self) -> Path:
        return self.out_dir / "results"


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
    values.setdefault("data_root", Path("data/raw"))
    return ExperimentConfig(**values)
