import tempfile
import unittest
from pathlib import Path

from ThaiSpoof.project.features import feature_groups_exist


class FeatureCacheTest(unittest.TestCase):
    def test_feature_groups_exist_when_all_expected_pickles_are_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in [
                "LFCC_Train_genuine_8.pkl",
                "LFCC_Train_spoof_8.pkl",
                "LFCC_Test_genuine_4.pkl",
                "LFCC_Test_spoof_4.pkl",
            ]:
                (root / name).write_bytes(b"cache")

            self.assertTrue(feature_groups_exist(root, "lfcc", 8, 8, 4, 4))

    def test_feature_groups_missing_when_count_does_not_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "LFCC_Train_genuine_7.pkl").write_bytes(b"cache")

            self.assertFalse(feature_groups_exist(root, "lfcc", 8, 8, 4, 4))


if __name__ == "__main__":
    unittest.main()
