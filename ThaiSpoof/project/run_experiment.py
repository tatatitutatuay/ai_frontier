from __future__ import annotations

import argparse
from collections import Counter
import logging
from pathlib import Path

from ThaiSpoof.project.config import CONFIG_FIELDS, ExperimentConfig, PRESETS, SUPPORTED_FEATURES, SUPPORTED_MODELS, resolve_experiment_config
from ThaiSpoof.project.dataset import collect_audio, read_manifest, split_balanced, split_balanced_by_spoof_attack, summarize_splits, write_manifest
from ThaiSpoof.project.features import feature_groups_exist, save_feature_groups
from ThaiSpoof.project.train import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a compact ThaiSpoof anti-spoofing experiment.")
    parser.add_argument("--data-root", type=Path, default=None, help="Folder containing ThaiSpoof audio files.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--stage", choices=["summary", "split", "extract", "train", "all"], default="all")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None, help="Hardware-sized experiment preset.")
    parser.add_argument("--config", type=Path, default=None, help="JSON config file.")
    parser.add_argument("--force-split", action="store_true", help="Rebuild the manifest even if it already exists.")
    parser.add_argument("--force-extract", action="store_true", help="Rebuild features even if cached files exist.")
    parser.add_argument("--feature", choices=sorted(SUPPORTED_FEATURES), default=None)
    parser.add_argument("--model", choices=sorted(SUPPORTED_MODELS), default=None)
    parser.add_argument("--spoof-attack", default=None, help="Use only one spoof attack/source, for example f0_10.")
    parser.add_argument("--spoof-attacks", default=None, help="Comma-separated spoof attacks/sources to include.")
    parser.add_argument("--train-genuine", type=int, default=None)
    parser.add_argument("--test-genuine", type=int, default=None)
    parser.add_argument("--train-spoof", type=int, default=None)
    parser.add_argument("--test-spoof", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--target-sr", type=int, default=None)
    parser.add_argument("--dim-x", type=int, default=None)
    parser.add_argument("--dim-y", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    return parser


def build_config_overrides(args: argparse.Namespace) -> dict:
    return {
        field: value
        for field in CONFIG_FIELDS
        if (value := getattr(args, field, None)) is not None
    }


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return resolve_experiment_config(
        preset_name=args.preset,
        config_file=args.config,
        overrides=build_config_overrides(args),
    )


def _manifest_paths_exist(splits) -> bool:
    return all(item.path.exists() for rows in splits.values() for item in rows)


def _configured_spoof_attacks(config: ExperimentConfig) -> tuple[str, ...] | None:
    if config.spoof_attacks:
        return config.spoof_attacks
    if config.spoof_attack:
        return (config.spoof_attack,)
    return None


def _manifest_matches_spoof_attacks(splits, spoof_attacks: tuple[str, ...] | None) -> bool:
    if not spoof_attacks:
        return True
    allowed_attacks = set(spoof_attacks)
    manifest_attacks = {
        item.attack_type
        for rows in splits.values()
        for item in rows
        if item.label == "spoof"
    }
    return bool(manifest_attacks) and manifest_attacks <= allowed_attacks


def _filter_items_for_spoof_attacks(items, spoof_attacks: tuple[str, ...] | None):
    if not spoof_attacks:
        return items
    allowed_attacks = set(spoof_attacks)
    return [
        item
        for item in items
        if item.label == "genuine" or item.attack_type in allowed_attacks
    ]


def create_or_load_splits(config: ExperimentConfig, force_create: bool = False):
    spoof_attacks = _configured_spoof_attacks(config)
    if config.manifest_path.exists() and not force_create:
        logging.info("Using existing manifest: %s", config.manifest_path)
        splits = read_manifest(config.manifest_path)
        if _manifest_paths_exist(splits) and _manifest_matches_spoof_attacks(splits, spoof_attacks):
            return splits
        logging.info("Existing manifest does not match current data/config; rebuilding it.")

    items = _filter_items_for_spoof_attacks(collect_audio(config.data_root), spoof_attacks)
    if spoof_attacks:
        logging.info("Filtering spoof class to attack/source(s): %s", ", ".join(sorted(spoof_attacks)))
    split_fn = split_balanced_by_spoof_attack if config.spoof_attacks else split_balanced
    if config.spoof_attacks:
        splits = split_fn(
            items,
            train_genuine=config.train_genuine,
            test_genuine=config.test_genuine,
            train_spoof=config.train_spoof,
            test_spoof=config.test_spoof,
            spoof_attacks=config.spoof_attacks,
            seed=config.seed,
        )
    else:
        splits = split_fn(
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


def log_dataset_summary(config: ExperimentConfig) -> None:
    spoof_attacks = _configured_spoof_attacks(config)
    items = _filter_items_for_spoof_attacks(collect_audio(config.data_root), spoof_attacks)
    label_counts = Counter(item.label for item in items)
    attack_counts = Counter(item.attack_type for item in items)
    logging.info("Data root: %s", config.data_root)
    if spoof_attacks:
        logging.info("Spoof attack/source filter: %s", ", ".join(sorted(spoof_attacks)))
    logging.info("Total audio files detected: %s", len(items))
    logging.info("Label counts: %s", dict(sorted(label_counts.items())))
    logging.info("Attack/source counts: %s", dict(sorted(attack_counts.items())))


def main() -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO)
    args = build_parser().parse_args()
    config = config_from_args(args)

    if args.stage == "summary":
        log_dataset_summary(config)
        return

    if args.stage == "split":
        create_or_load_splits(config, force_create=True or args.force_split)
        return

    if args.stage in {"extract", "all"}:
        splits = create_or_load_splits(config, force_create=args.force_split)
        if not args.force_extract and feature_groups_exist(
            config.feature_dir,
            config.feature,
            train_genuine=len(splits["train_genuine"]),
            train_spoof=len(splits["train_spoof"]),
            test_genuine=len(splits["test_genuine"]),
            test_spoof=len(splits["test_spoof"]),
        ):
            logging.info("Using cached %s features in %s", config.feature.upper(), config.feature_dir)
        else:
            save_feature_groups(splits, config.feature_dir, config.feature, config.target_sr)

    if args.stage in {"train", "all"}:
        metrics_path = run_training(config)
        logging.info("Training complete. Metrics: %s", metrics_path)


if __name__ == "__main__":
    main()
