import unittest

from ThaiSpoof.project.metrics import calculate_binary_metrics, compute_eer


class MetricsTest(unittest.TestCase):
    def test_perfect_scores_have_zero_eer_and_full_metrics(self):
        y_true = [0, 0, 1, 1]
        scores = [0.05, 0.2, 0.8, 0.95]

        metrics = calculate_binary_metrics(y_true, scores)

        self.assertEqual(metrics.tn, 2)
        self.assertEqual(metrics.tp, 2)
        self.assertEqual(metrics.fp, 0)
        self.assertEqual(metrics.fn, 0)
        self.assertAlmostEqual(metrics.accuracy, 1.0)
        self.assertAlmostEqual(metrics.f1, 1.0)
        self.assertAlmostEqual(metrics.eer, 0.0)

    def test_threshold_confusion_counts_use_spoof_as_positive_class(self):
        y_true = [0, 0, 1, 1]
        scores = [0.7, 0.1, 0.4, 0.9]

        metrics = calculate_binary_metrics(y_true, scores, threshold=0.5)

        self.assertEqual((metrics.tn, metrics.fp, metrics.fn, metrics.tp), (1, 1, 1, 1))
        self.assertAlmostEqual(metrics.precision, 0.5)
        self.assertAlmostEqual(metrics.recall, 0.5)

    def test_eer_is_between_zero_and_one_for_tied_scores(self):
        eer, threshold = compute_eer([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5])

        self.assertGreaterEqual(eer, 0.0)
        self.assertLessEqual(eer, 1.0)
        self.assertEqual(threshold, 0.5)


if __name__ == "__main__":
    unittest.main()
