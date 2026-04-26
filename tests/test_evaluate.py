import pickle
import tempfile
import unittest
import csv
from pathlib import Path

import numpy as np

from ThaiSpoof.project.evaluate import evaluate_model_on_attack_from_feature_dir, evaluate_model_on_feature_dir


class FakeModel:
    def __init__(self, scores=None):
        self.scores = scores or [0.1, 0.2, 0.8, 0.9]

    def predict(self, x, batch_size=64, verbose=0):
        return np.array(self.scores[: len(x)], dtype=np.float32)[:, None]


def write_pickle(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)


def write_index(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "attack_type", "frames", "dims"])
        writer.writeheader()
        writer.writerows(rows)


class EvaluateTest(unittest.TestCase):
    def test_evaluate_model_on_feature_dir_reports_metrics_and_mean_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            feature_dir = Path(tmp)
            genuine = [np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)]
            spoof = [np.full((2, 2), 2, dtype=np.float32), np.full((2, 2), 3, dtype=np.float32)]
            write_pickle(feature_dir / "LFCC_Test_genuine_2.pkl", genuine)
            write_pickle(feature_dir / "LFCC_Test_spoof_2.pkl", spoof)

            result = evaluate_model_on_feature_dir(
                FakeModel(),
                feature_dir=feature_dir,
                feature="lfcc",
                dim_x=2,
                dim_y=2,
                split="Test",
                name="fake_eval",
            )

        self.assertEqual(result.name, "fake_eval")
        self.assertEqual(result.genuine_count, 2)
        self.assertEqual(result.spoof_count, 2)
        self.assertEqual((result.metrics.tn, result.metrics.fp, result.metrics.fn, result.metrics.tp), (2, 0, 0, 2))
        self.assertAlmostEqual(result.mean_genuine_score, 0.15)
        self.assertAlmostEqual(result.mean_spoof_score, 0.85)

    def test_evaluate_model_on_attack_from_feature_dir_uses_indexed_spoof_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            feature_dir = Path(tmp)
            genuine = [np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)]
            spoof = [
                np.full((2, 2), 2, dtype=np.float32),
                np.full((2, 2), 3, dtype=np.float32),
                np.full((2, 2), 4, dtype=np.float32),
            ]
            write_pickle(feature_dir / "LFCC_Test_genuine_2.pkl", genuine)
            write_pickle(feature_dir / "LFCC_Test_spoof_3.pkl", spoof)
            write_index(
                feature_dir / "INDEX_Test_spoof_3.csv",
                [
                    {"path": "a.wav", "label": "spoof", "attack_type": "tts", "frames": 2, "dims": 2},
                    {"path": "b.wav", "label": "spoof", "attack_type": "f0_10", "frames": 2, "dims": 2},
                    {"path": "c.wav", "label": "spoof", "attack_type": "tts", "frames": 2, "dims": 2},
                ],
            )

            result = evaluate_model_on_attack_from_feature_dir(
                FakeModel(scores=[0.1, 0.2, 0.8, 0.9]),
                feature_dir=feature_dir,
                feature="lfcc",
                attack_type="tts",
                dim_x=2,
                dim_y=2,
                split="Test",
            )

        self.assertEqual(result.name, "tts")
        self.assertEqual(result.genuine_count, 2)
        self.assertEqual(result.spoof_count, 2)
        self.assertEqual((result.metrics.tn, result.metrics.fp, result.metrics.fn, result.metrics.tp), (2, 0, 0, 2))


if __name__ == "__main__":
    unittest.main()
