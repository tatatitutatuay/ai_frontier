import tempfile
import unittest
from pathlib import Path

from ThaiSpoof.project.config import ExperimentConfig


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


if __name__ == "__main__":
    unittest.main()
