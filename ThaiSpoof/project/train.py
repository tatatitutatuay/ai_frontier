from __future__ import annotations

import csv
import io
import pickle
from pathlib import Path

from ThaiSpoof.project.config import ExperimentConfig
from ThaiSpoof.project.features import pad_or_repeat
from ThaiSpoof.project.metrics import BinaryMetrics, calculate_binary_metrics
from ThaiSpoof.project.models import build_model


class _NumpyCoreCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_pickle_list(path: Path):
    with Path(path).open("rb") as handle:
        with io.BufferedReader(handle) as buffered:
            obj = _NumpyCoreCompatUnpickler(buffered).load()
    if not isinstance(obj, (list, tuple)):
        raise ValueError(f"expected a list of feature matrices in {path}")
    return list(obj)


def _find_feature_pickle(feature_dir: Path, feature: str, split: str, label: str) -> Path:
    pattern = f"{feature.upper()}_{split}_{label}_*.pkl"
    matches = sorted(Path(feature_dir).glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no feature file matched {pattern} in {feature_dir}")
    return matches[-1]


def load_feature_groups(feature_dir: Path, feature: str) -> tuple[list, list, list, list]:
    train_g = load_pickle_list(_find_feature_pickle(feature_dir, feature, "Train", "genuine"))
    train_s = load_pickle_list(_find_feature_pickle(feature_dir, feature, "Train", "spoof"))
    test_g = load_pickle_list(_find_feature_pickle(feature_dir, feature, "Test", "genuine"))
    test_s = load_pickle_list(_find_feature_pickle(feature_dir, feature, "Test", "spoof"))
    return train_g, train_s, test_g, test_s


def _prepare_xy(genuine, spoof, dim_x: int, dim_y: int):
    import numpy as np

    rows = [pad_or_repeat(item, dim_x, dim_y) for item in genuine + spoof]
    x = np.stack(rows, axis=0)[..., None].astype(np.float32)
    y = np.concatenate(
        [
            np.zeros(len(genuine), dtype=np.int32),
            np.ones(len(spoof), dtype=np.int32),
        ]
    )
    return x, y


def _split_train_validation(x, y, validation_fraction: float, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    for label in [0, 1]:
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        val_count = max(1, int(round(len(idx) * validation_fraction)))
        val_idx.extend(idx[:val_count])
        train_idx.extend(idx[val_count:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def _metrics_row(split: str, metrics: BinaryMetrics) -> dict[str, str | int | float]:
    return {
        "split": split,
        "tn": metrics.tn,
        "fp": metrics.fp,
        "fn": metrics.fn,
        "tp": metrics.tp,
        "accuracy": metrics.accuracy,
        "balanced_accuracy": metrics.balanced_accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "eer": metrics.eer,
        "eer_threshold": metrics.eer_threshold,
    }


def _evaluate_model(model, x, y) -> BinaryMetrics:
    scores = model.predict(x, batch_size=64, verbose=0).reshape(-1).tolist()
    return calculate_binary_metrics(y.astype(int).tolist(), [float(score) for score in scores])


def _write_metrics(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_history(path: Path, history) -> None:
    keys = list(history.history.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", *keys])
        for idx in range(len(history.history[keys[0]])):
            writer.writerow([idx + 1, *[history.history[key][idx] for key in keys]])


def run_training(config: ExperimentConfig) -> Path:
    import random
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    random.seed(config.seed)
    np.random.seed(config.seed)
    tf.keras.utils.set_random_seed(config.seed)

    train_g, train_s, test_g, test_s = load_feature_groups(config.feature_dir, config.feature)
    x_all, y_all = _prepare_xy(train_g, train_s, config.dim_x, config.dim_y)
    x_train, y_train, x_val, y_val = _split_train_validation(
        x_all,
        y_all,
        config.validation_fraction,
        config.seed,
    )
    x_test, y_test = _prepare_xy(test_g, test_s, config.dim_x, config.dim_y)

    config.model_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config.model, (config.dim_x, config.dim_y, 1), config.learning_rate)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=config.patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=max(2, config.patience // 2), factor=0.5),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=2,
        shuffle=True,
    )

    model_path = config.model_dir / f"{config.feature}_{config.model}.keras"
    model.save(model_path)
    _write_history(config.results_dir / f"{config.feature}_{config.model}_history.csv", history)

    rows = [
        _metrics_row("train", _evaluate_model(model, x_train, y_train)),
        _metrics_row("validation", _evaluate_model(model, x_val, y_val)),
        _metrics_row("test", _evaluate_model(model, x_test, y_test)),
    ]
    metrics_path = config.results_dir / f"{config.feature}_{config.model}_metrics.csv"
    _write_metrics(metrics_path, rows)

    with (config.results_dir / f"{config.feature}_{config.model}_summary.md").open("w", encoding="utf-8") as handle:
        handle.write(f"# ThaiSpoof {config.feature.upper()} {config.model} Summary\n\n")
        handle.write(f"- Model file: `{model_path}`\n")
        handle.write(f"- Metrics file: `{metrics_path}`\n")
        handle.write(f"- Train samples: {len(y_train)}\n")
        handle.write(f"- Validation samples: {len(y_val)}\n")
        handle.write(f"- Test samples: {len(y_test)}\n")

    return metrics_path
