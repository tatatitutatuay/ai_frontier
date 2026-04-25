from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
