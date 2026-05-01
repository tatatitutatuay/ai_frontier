import unittest

import numpy as np

from ThaiSpoof.project.train import _evaluate_attack_metrics


class FakeModel:
    def predict(self, x, batch_size=64, verbose=0):
        scores = [0.1, 0.2, 0.8, 0.9]
        return np.array(scores[: len(x)], dtype=np.float32)[:, None]


class TrainAttackMetricsTest(unittest.TestCase):
    def test_evaluate_attack_metrics_reports_each_spoof_source(self):
        genuine = [
            np.zeros((2, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
        ]
        spoof = [
            np.full((2, 2), 2, dtype=np.float32),
            np.full((2, 2), 3, dtype=np.float32),
            np.full((2, 2), 4, dtype=np.float32),
        ]

        rows = _evaluate_attack_metrics(
            FakeModel(),
            genuine=genuine,
            spoof=spoof,
            spoof_attack_types=["vaja", "f0_40", "vaja"],
            dim_x=2,
            dim_y=2,
            split="test",
        )

        by_attack = {row["attack_type"]: row for row in rows}
        self.assertEqual(set(by_attack), {"f0_40", "vaja"})
        self.assertEqual(by_attack["f0_40"]["genuine_count"], 2)
        self.assertEqual(by_attack["f0_40"]["spoof_count"], 1)
        self.assertEqual(by_attack["vaja"]["genuine_count"], 2)
        self.assertEqual(by_attack["vaja"]["spoof_count"], 2)
        self.assertEqual(by_attack["vaja"]["f1"], 1.0)

    def test_evaluate_attack_metrics_requires_matching_index_rows(self):
        with self.assertRaises(ValueError):
            _evaluate_attack_metrics(
                FakeModel(),
                genuine=[np.zeros((2, 2), dtype=np.float32)],
                spoof=[np.ones((2, 2), dtype=np.float32)],
                spoof_attack_types=[],
                dim_x=2,
                dim_y=2,
            )


if __name__ == "__main__":
    unittest.main()
