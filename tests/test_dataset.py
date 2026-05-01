import tempfile
import unittest
from pathlib import Path

from ThaiSpoof.project.dataset import (
    collect_audio,
    read_manifest,
    split_balanced,
    split_balanced_by_spoof_attack,
    summarize_splits,
    utterance_group_id,
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
            touch(root / "Corpus-Spoof-genuine" / "genuine" / "G1" / "thai0001.wav")
            touch(root / "Corpus-Spoof-VAJA" / "Train" / "thai_000001.wav")
            touch(root / "Corpus-Spoof-F0_40" / "F0_40" / "f0_40_1" / "f0_40_thai0001.wav")
            touch(root / "Corpus-Spoof-F0_160_new" / "f0_160_1" / "f0_160_thai0001.wav")

            items = collect_audio(root)

        labels = sorted(item.label for item in items)
        attacks = sorted(item.attack_type for item in items)
        self.assertEqual(labels, ["genuine", "spoof", "spoof", "spoof"])
        self.assertIn("corpus_spoof_vaja", attacks)
        self.assertIn("f0_40", attacks)
        self.assertIn("f0_160", attacks)

    def test_collect_audio_detects_f0_attack_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            touch(root / "F0_10" / "f0_10_1" / "f0_10_thai0001.wav")
            touch(root / "f0_10_2" / "f0_10_thai0002.wav")

            items = collect_audio(root)

        self.assertEqual(len(items), 2)
        self.assertEqual([item.label for item in items], ["spoof", "spoof"])
        self.assertEqual({item.attack_type for item in items}, {"f0_10"})

    def test_collect_audio_skips_macos_metadata_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            touch(root / "Corpus-Spoof-genuine" / "genuine" / "G4" / "thai3094.wav")
            touch(root / "Corpus-Spoof-genuine" / "__MACOSX" / "genuine" / "G4" / "._thai3094.wav")

            items = collect_audio(root)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].path.name, "thai3094.wav")

    def test_utterance_group_id_matches_genuine_and_spoof_variants(self):
        paths = [
            Path("thai0001.wav"),
            Path("thai_000001.wav"),
            Path("f0_40_thai0001.wav"),
            Path("f0_160_thai0001.wav"),
        ]

        self.assertEqual({utterance_group_id(path) for path in paths}, {"thai0001"})

    def test_split_balanced_keeps_related_utterances_on_same_side(self):
        items = [
            type("Item", (), {"label": "genuine", "path": Path("thai0001.wav"), "attack_type": "genuine"})(),
            type("Item", (), {"label": "spoof", "path": Path("thai_000001.wav"), "attack_type": "corpus_spoof_vaja"})(),
            type("Item", (), {"label": "spoof", "path": Path("f0_40_thai0001.wav"), "attack_type": "f0_40"})(),
            type("Item", (), {"label": "genuine", "path": Path("thai0002.wav"), "attack_type": "genuine"})(),
            type("Item", (), {"label": "spoof", "path": Path("thai_000002.wav"), "attack_type": "corpus_spoof_vaja"})(),
            type("Item", (), {"label": "spoof", "path": Path("f0_40_thai0002.wav"), "attack_type": "f0_40"})(),
        ]

        splits = split_balanced_by_spoof_attack(
            items,
            train_genuine=1,
            test_genuine=1,
            train_spoof=2,
            test_spoof=2,
            spoof_attacks=["corpus_spoof_vaja", "f0_40"],
            seed=3,
        )

        train_groups = {
            utterance_group_id(item.path)
            for split in ("train_genuine", "train_spoof")
            for item in splits[split]
        }
        test_groups = {
            utterance_group_id(item.path)
            for split in ("test_genuine", "test_spoof")
            for item in splits[split]
        }
        self.assertTrue(train_groups.isdisjoint(test_groups))


if __name__ == "__main__":
    unittest.main()
