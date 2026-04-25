import tempfile
import unittest
from pathlib import Path

from ThaiSpoof.project.dataset import (
    collect_audio,
    read_manifest,
    split_balanced,
    summarize_splits,
    write_manifest,
)


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


class DatasetTest(unittest.TestCase):
    def test_collect_audio_infers_genuine_and_spoof_from_path_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            touch(root / "speaker_a" / "genuine" / "a.wav")
            touch(root / "speaker_b" / "tts_vaja_spoof" / "b.WAV")
            touch(root / "speaker_c" / "Pitch Shift Spoofed Audio" / "c.wav")
            touch(root / "speaker_c" / "not_audio" / "note.txt")

            items = collect_audio(root)

        labels = sorted(item.label for item in items)
        self.assertEqual(labels, ["genuine", "spoof", "spoof"])
        self.assertEqual(len(items), 3)

    def test_split_balanced_has_no_overlap_and_scales_down_when_needed(self):
        items = []
        for i in range(4):
            items.append(("genuine", f"g{i}.wav"))
        for i in range(4):
            items.append(("spoof", f"s{i}.wav"))

        splits = split_balanced(
            [
                # use tiny stand-ins to avoid filesystem dependency
                type("Item", (), {"label": label, "path": Path(path), "attack_type": label})()
                for label, path in items
            ],
            train_genuine=3,
            test_genuine=3,
            train_spoof=3,
            test_spoof=3,
            seed=7,
        )

        self.assertEqual(len(splits["train_genuine"]), 2)
        self.assertEqual(len(splits["test_genuine"]), 2)
        self.assertEqual(len(splits["train_spoof"]), 2)
        self.assertEqual(len(splits["test_spoof"]), 2)

        train_paths = {item.path for item in splits["train_genuine"] + splits["train_spoof"]}
        test_paths = {item.path for item in splits["test_genuine"] + splits["test_spoof"]}
        self.assertTrue(train_paths.isdisjoint(test_paths))

    def test_manifest_roundtrip_preserves_split_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            touch(root / "genuine" / "a.wav")
            touch(root / "spoof" / "b.wav")
            items = collect_audio(root)
            splits = {
                "train_genuine": [item for item in items if item.label == "genuine"],
                "train_spoof": [item for item in items if item.label == "spoof"],
                "test_genuine": [],
                "test_spoof": [],
            }

            manifest = root / "manifest.csv"
            write_manifest(splits, manifest)
            loaded = read_manifest(manifest)

        self.assertEqual(summarize_splits(loaded)["train_genuine"], 1)
        self.assertEqual(summarize_splits(loaded)["train_spoof"], 1)

    def test_collect_audio_detects_current_workspace_layout_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            touch(root / "genuine" / "G1" / "thai0001.wav")
            touch(root / "Corpus-Spoof-VAJA" / "Train" / "thai_000001.wav")

            items = collect_audio(root)

        labels = sorted(item.label for item in items)
        attacks = sorted(item.attack_type for item in items)
        self.assertEqual(labels, ["genuine", "spoof"])
        self.assertIn("corpus_spoof_vaja", attacks)


if __name__ == "__main__":
    unittest.main()
