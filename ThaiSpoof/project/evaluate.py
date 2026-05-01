from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from ThaiSpoof.project.config import ExperimentConfig
from ThaiSpoof.project.metrics import BinaryMetrics, calculate_binary_metrics
from ThaiSpoof.project.train import _find_feature_pickle, _prepare_xy, load_pickle_list


@dataclass(frozen=True)
class EvaluationResult:
    name: str
    split: str
    genuine_count: int
    spoof_count: int
    metrics: BinaryMetrics
    mean_genuine_score: float
    mean_spoof_score: float


def evaluate_model_on_feature_dir(
    model,
    feature_dir: Path,
    feature: str,
    dim_x: int,
    dim_y: int,
    split: str = "Test",
    batch_size: int = 64,
    name: str = "evaluation",
) -> EvaluationResult:
    genuine = load_pickle_list(_find_feature_pickle(feature_dir, feature, split, "genuine"))
    spoof = load_pickle_list(_find_feature_pickle(feature_dir, feature, split, "spoof"))
    x, y = _prepare_xy(genuine, spoof, dim_x, dim_y)
    scores = model.predict(x, batch_size=batch_size, verbose=0).reshape(-1).astype(float).tolist()
    metrics = calculate_binary_metrics(y.astype(int).tolist(), scores)
    genuine_scores = scores[: len(genuine)]
    spoof_scores = scores[len(genuine) :]

    return EvaluationResult(
        name=name,
        split=split,
        genuine_count=len(genuine),
        spoof_count=len(spoof),
        metrics=metrics,
        mean_genuine_score=sum(genuine_scores) / len(genuine_scores),
        mean_spoof_score=sum(spoof_scores) / len(spoof_scores),
    )


def _find_index_csv(feature_dir: Path, split: str, label: str) -> Path:
    pattern = f"INDEX_{split}_{label}_*.csv"
    matches = sorted(Path(feature_dir).glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no index file matched {pattern} in {feature_dir}")
    return matches[-1]


def _read_index_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evaluate_model_on_attack_from_feature_dir(
    model,
    feature_dir: Path,
    feature: str,
    attack_type: str,
    dim_x: int,
    dim_y: int,
    split: str = "Test",
    batch_size: int = 64,
) -> EvaluationResult:
    genuine = load_pickle_list(_find_feature_pickle(feature_dir, feature, split, "genuine"))
    spoof = load_pickle_list(_find_feature_pickle(feature_dir, feature, split, "spoof"))
    spoof_rows = _read_index_rows(_find_index_csv(feature_dir, split, "spoof"))
    selected_spoof = [
        feature_row
        for feature_row, index_row in zip(spoof, spoof_rows)
        if index_row.get("attack_type") == attack_type
    ]
    if not selected_spoof:
        raise ValueError(f"no spoof features found for attack/source: {attack_type}")

    x, y = _prepare_xy(genuine, selected_spoof, dim_x, dim_y)
    scores = model.predict(x, batch_size=batch_size, verbose=0).reshape(-1).astype(float).tolist()
    metrics = calculate_binary_metrics(y.astype(int).tolist(), scores)
    genuine_scores = scores[: len(genuine)]
    spoof_scores = scores[len(genuine) :]
    return EvaluationResult(
        name=attack_type,
        split=split,
        genuine_count=len(genuine),
        spoof_count=len(selected_spoof),
        metrics=metrics,
        mean_genuine_score=sum(genuine_scores) / len(genuine_scores),
        mean_spoof_score=sum(spoof_scores) / len(spoof_scores),
    )


def evaluation_row(result: EvaluationResult, model_path: Path, feature_dir: Path) -> dict[str, str | int | float]:
    return {
        "name": result.name,
        "source_model": str(model_path),
        "eval_features": str(feature_dir),
        "split": result.split,
        "genuine_count": result.genuine_count,
        "spoof_count": result.spoof_count,
        "tn": result.metrics.tn,
        "fp": result.metrics.fp,
        "fn": result.metrics.fn,
        "tp": result.metrics.tp,
        "accuracy": result.metrics.accuracy,
        "balanced_accuracy": result.metrics.balanced_accuracy,
        "precision": result.metrics.precision,
        "recall": result.metrics.recall,
        "f1": result.metrics.f1,
        "eer": result.metrics.eer,
        "eer_threshold": result.metrics.eer_threshold,
        "mean_genuine_score": result.mean_genuine_score,
        "mean_spoof_score": result.mean_spoof_score,
    }


def write_evaluation_outputs(
    result: EvaluationResult,
    output_dir: Path,
    model_path: Path,
    feature_dir: Path,
    stem: str,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    row = evaluation_row(result, model_path, feature_dir)
    metrics_path = output_dir / f"{stem}_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    summary_path = output_dir / f"{stem}_summary.md"
    summary_path.write_text(
        f"# {result.name}\n\n"
        f"- Source model: `{model_path}`\n"
        f"- Evaluation features: `{feature_dir}`\n"
        f"- Split: {result.split}\n"
        f"- Genuine samples: {result.genuine_count}\n"
        f"- Spoof samples: {result.spoof_count}\n"
        f"- Accuracy: {result.metrics.accuracy:.3f}\n"
        f"- Precision: {result.metrics.precision:.3f}\n"
        f"- Recall: {result.metrics.recall:.3f}\n"
        f"- F1: {result.metrics.f1:.3f}\n"
        f"- EER: {result.metrics.eer:.3f}\n"
        f"- Confusion matrix: TN={result.metrics.tn}, FP={result.metrics.fp}, "
        f"FN={result.metrics.fn}, TP={result.metrics.tp}\n"
        f"- Mean genuine score: {result.mean_genuine_score:.3f}\n"
        f"- Mean spoof score: {result.mean_spoof_score:.3f}\n",
        encoding="utf-8",
    )
    return metrics_path


def simple_evaluation_row(result: EvaluationResult) -> dict[str, str | int | float]:
    return {
        "name": result.name,
        "split": result.split,
        "genuine_count": result.genuine_count,
        "spoof_count": result.spoof_count,
        "tn": result.metrics.tn,
        "fp": result.metrics.fp,
        "fn": result.metrics.fn,
        "tp": result.metrics.tp,
        "accuracy": result.metrics.accuracy,
        "balanced_accuracy": result.metrics.balanced_accuracy,
        "precision": result.metrics.precision,
        "recall": result.metrics.recall,
        "f1": result.metrics.f1,
        "eer": result.metrics.eer,
        "eer_threshold": result.metrics.eer_threshold,
        "mean_genuine_score": result.mean_genuine_score,
        "mean_spoof_score": result.mean_spoof_score,
    }


def write_evaluation_table(results: list[EvaluationResult], path: Path) -> Path:
    if not results:
        raise ValueError("at least one evaluation result is required")

    rows = [simple_evaluation_row(result) for result in results]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _model_path(config: ExperimentConfig) -> Path:
    return config.model_dir / f"{config.feature}_{config.model}.keras"


def _configured_or_indexed_attacks(config: ExperimentConfig) -> list[str]:
    if config.spoof_attacks:
        return list(config.spoof_attacks)
    if config.spoof_attack:
        return [config.spoof_attack]

    index_path = _find_index_csv(config.feature_dir, "Test", "spoof")
    rows = _read_index_rows(index_path)
    return sorted({row.get("attack_type") or "spoof" for row in rows})


def evaluate_finished_model(config: ExperimentConfig, model_loader=None) -> Path:
    model_path = _model_path(config)
    if not model_path.exists():
        raise FileNotFoundError(f"trained model does not exist: {model_path}")

    if model_loader is None:
        import tensorflow as tf

        model_loader = tf.keras.models.load_model

    model = model_loader(str(model_path))
    results = [
        evaluate_model_on_feature_dir(
            model,
            feature_dir=config.feature_dir,
            feature=config.feature,
            dim_x=config.dim_x,
            dim_y=config.dim_y,
            split="Test",
            name="overall",
        )
    ]
    for attack_type in _configured_or_indexed_attacks(config):
        results.append(
            evaluate_model_on_attack_from_feature_dir(
                model,
                feature_dir=config.feature_dir,
                feature=config.feature,
                attack_type=attack_type,
                dim_x=config.dim_x,
                dim_y=config.dim_y,
                split="Test",
            )
        )

    output_path = config.results_dir / f"{config.feature}_{config.model}_evaluate_metrics.csv"
    return write_evaluation_table(results, output_path)
