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
                spoof_attack="F0-10",
            )

        self.assertEqual(cfg.feature, "lfcc")
        self.assertEqual(cfg.model, "small_cnn")
        self.assertEqual(cfg.spoof_attack, "f0_10")

    def test_normalizes_multiple_spoof_attacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ExperimentConfig(
                data_root=Path(tmp) / "data",
                out_dir=Path(tmp) / "out",
                spoof_attacks=["F0-10", "Corpus Spoof VAJA"],
            )

        self.assertEqual(cfg.spoof_attacks, ("f0_10", "corpus_spoof_vaja"))

    def test_rejects_single_and_multiple_spoof_attack_filters_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                ExperimentConfig(
                    data_root=Path(tmp) / "data",
                    out_dir=Path(tmp) / "out",
                    spoof_attack="f0_10",
                    spoof_attacks=["corpus_spoof_vaja"],
                )

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

    def test_config_defaults_to_local_data_folder(self):
        cfg = resolve_experiment_config(preset_name="smoke")

        self.assertEqual(cfg.data_root, Path("data"))

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
