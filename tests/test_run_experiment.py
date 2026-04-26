import unittest
from pathlib import Path

from ThaiSpoof.project.config import ExperimentConfig
from ThaiSpoof.project.run_experiment import build_config_overrides, build_parser, create_or_load_splits


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


class RunExperimentCliTest(unittest.TestCase):
    def test_parser_accepts_summary_stage_and_preset(self):
        args = build_parser().parse_args(
            [
                "--data-root",
                ".",
                "--preset",
                "smoke",
                "--stage",
                "summary",
            ]
        )

        self.assertEqual(args.data_root, Path("."))
        self.assertEqual(args.preset, "smoke")
        self.assertEqual(args.stage, "summary")

    def test_build_config_overrides_ignores_non_config_arguments_and_none_values(self):
        args = build_parser().parse_args(
            [
                "--data-root",
                ".",
                "--preset",
                "smoke",
                "--stage",
                "summary",
                "--batch-size",
                "12",
            ]
        )

        overrides = build_config_overrides(args)

        self.assertEqual(overrides, {"data_root": Path("."), "batch_size": 12})

    def test_parser_accepts_spoof_attack_filter(self):
        args = build_parser().parse_args(
            [
                "--data-root",
                ".",
                "--stage",
                "summary",
                "--spoof-attack",
                "F0-10",
            ]
        )

        overrides = build_config_overrides(args)

        self.assertEqual(overrides["spoof_attack"], "F0-10")

    def test_parser_accepts_multiple_spoof_attack_filter(self):
        args = build_parser().parse_args(
            [
                "--data-root",
                ".",
                "--stage",
                "summary",
                "--spoof-attacks",
                "F0-10,Corpus-Spoof-VAJA",
            ]
        )

        overrides = build_config_overrides(args)

        self.assertEqual(overrides["spoof_attacks"], "F0-10,Corpus-Spoof-VAJA")

    def test_create_or_load_splits_rebuilds_manifest_when_paths_are_stale(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data" / "raw"
            out_dir = root / "runs"
            touch(data_root / "genuine" / "a.wav")
            touch(data_root / "genuine" / "b.wav")
            touch(data_root / "Corpus-Spoof-VAJA" / "Train" / "c.wav")
            touch(data_root / "Corpus-Spoof-VAJA" / "Train" / "d.wav")
            out_dir.mkdir()
            (out_dir / "manifest.csv").write_text(
                "split,label,attack_type,path\n"
                "train_genuine,genuine,genuine,old/missing.wav\n",
                encoding="utf-8",
            )
            cfg = ExperimentConfig(
                data_root=data_root,
                out_dir=out_dir,
                train_genuine=1,
                train_spoof=1,
                test_genuine=1,
                test_spoof=1,
            )

            splits = create_or_load_splits(cfg)
            all_paths = [item.path for rows in splits.values() for item in rows]
            self.assertTrue(all_paths)
            self.assertTrue(all(path.exists() for path in all_paths))

    def test_create_or_load_splits_filters_requested_spoof_attack(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data" / "raw"
            out_dir = root / "runs"
            touch(data_root / "genuine" / "a.wav")
            touch(data_root / "genuine" / "b.wav")
            touch(data_root / "Corpus-Spoof-VAJA" / "Train" / "vaja_a.wav")
            touch(data_root / "Corpus-Spoof-VAJA" / "Train" / "vaja_b.wav")
            touch(data_root / "F0_10" / "f0_10_1" / "f0_a.wav")
            touch(data_root / "F0_10" / "f0_10_1" / "f0_b.wav")
            cfg = ExperimentConfig(
                data_root=data_root,
                out_dir=out_dir,
                spoof_attack="f0_10",
                train_genuine=1,
                train_spoof=1,
                test_genuine=1,
                test_spoof=1,
            )

            splits = create_or_load_splits(cfg)
            spoof_attacks = {
                item.attack_type
                for split_name, rows in splits.items()
                if "spoof" in split_name
                for item in rows
            }

        self.assertEqual(spoof_attacks, {"f0_10"})

    def test_create_or_load_splits_filters_requested_spoof_attacks(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data" / "raw"
            out_dir = root / "runs"
            touch(data_root / "genuine" / "a.wav")
            touch(data_root / "genuine" / "b.wav")
            for idx in range(4):
                touch(data_root / "Corpus-Spoof-VAJA" / "Train" / f"vaja_{idx}.wav")
                touch(data_root / "F0_10" / "f0_10_1" / f"f0_{idx}.wav")
                touch(data_root / "pitch_shift" / f"p_{idx}.wav")
            cfg = ExperimentConfig(
                data_root=data_root,
                out_dir=out_dir,
                spoof_attacks=["f0_10", "corpus_spoof_vaja"],
                train_genuine=1,
                train_spoof=4,
                test_genuine=1,
                test_spoof=4,
            )

            splits = create_or_load_splits(cfg)
            attack_counts = {
                split_name: {
                    attack: sum(1 for item in rows if item.attack_type == attack)
                    for attack in {"f0_10", "corpus_spoof_vaja", "pitch_shift"}
                }
                for split_name, rows in splits.items()
                if "spoof" in split_name
            }

        self.assertEqual(attack_counts["train_spoof"], {"f0_10": 2, "corpus_spoof_vaja": 2, "pitch_shift": 0})
        self.assertEqual(attack_counts["test_spoof"], {"f0_10": 2, "corpus_spoof_vaja": 2, "pitch_shift": 0})


if __name__ == "__main__":
    unittest.main()
