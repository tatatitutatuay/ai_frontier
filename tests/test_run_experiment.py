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


if __name__ == "__main__":
    unittest.main()
