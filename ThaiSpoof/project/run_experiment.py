from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ThaiSpoof.project.config import ExperimentConfig, SUPPORTED_FEATURES, SUPPORTED_MODELS
from ThaiSpoof.project.dataset import collect_audio, read_manifest, split_balanced, summarize_splits, write_manifest
from ThaiSpoof.project.features import save_feature_groups
from ThaiSpoof.project.train import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a compact ThaiSpoof anti-spoofing experiment.")
    parser.add_argument("--data-root", type=Path, required=True, help="Folder containing ThaiSpoof audio files.")
    parser.add_argument("--out-dir", type=Path, default=Path("ThaiSpoof/runs/thaispoof_lfcc"))
    parser.add_argument("--stage", choices=["split", "extract", "train", "all"], default="all")
    parser.add_argument("--feature", choices=sorted(SUPPORTED_FEATURES), default="lfcc")
    parser.add_argument("--model", choices=sorted(SUPPORTED_MODELS), default="small_cnn")
    parser.add_argument("--train-genuine", type=int, default=3000)
    parser.add_argument("--test-genuine", type=int, default=1500)
    parser.add_argument("--train-spoof", type=int, default=3000)
    parser.add_argument("--test-spoof", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--dim-x", type=int, default=256)
    parser.add_argument("--dim-y", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        data_root=args.data_root,
        out_dir=args.out_dir,
        feature=args.feature,
        model=args.model,
        train_genuine=args.train_genuine,
        test_genuine=args.test_genuine,
        train_spoof=args.train_spoof,
        test_spoof=args.test_spoof,
        seed=args.seed,
        target_sr=args.target_sr,
        dim_x=args.dim_x,
        dim_y=args.dim_y,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_fraction=args.validation_fraction,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )


def create_or_load_splits(config: ExperimentConfig, force_create: bool = False):
    if config.manifest_path.exists() and not force_create:
        logging.info("Using existing manifest: %s", config.manifest_path)
        return read_manifest(config.manifest_path)

    items = collect_audio(config.data_root)
    splits = split_balanced(
        items,
        train_genuine=config.train_genuine,
        test_genuine=config.test_genuine,
        train_spoof=config.train_spoof,
        test_spoof=config.test_spoof,
        seed=config.seed,
    )
    write_manifest(splits, config.manifest_path)
    logging.info("Wrote manifest to %s", config.manifest_path)
    logging.info("Split summary: %s", summarize_splits(splits))
    return splits


def main() -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO)
    args = build_parser().parse_args()
    config = config_from_args(args)

    if args.stage == "split":
        create_or_load_splits(config, force_create=True)
        return

    if args.stage in {"extract", "all"}:
        splits = create_or_load_splits(config)
        save_feature_groups(splits, config.feature_dir, config.feature, config.target_sr)

    if args.stage in {"train", "all"}:
        metrics_path = run_training(config)
        logging.info("Training complete. Metrics: %s", metrics_path)


if __name__ == "__main__":
    main()
