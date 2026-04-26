from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a"}

CLASS_ALIASES = {
    "genuine": {"genuine", "bona_fide", "bonafide", "real", "human", "bona"},
    "spoof": {
        "spoof",
        "spoofed",
        "synthetic",
        "fake",
        "mms_spoof",
        "mms_spoof_root",
        "tts_vaja_spoof",
        "tts",
        "vaja",
        "f0",
        "pitchshift",
        "pitch_shift",
        "speedchange",
        "speed_change",
    },
}


@dataclass(frozen=True)
class AudioItem:
    path: Path
    label: str
    attack_type: str


def normalize_token(value: str) -> str:
    return value.strip().replace("-", "_").replace(" ", "_").lower()


def _canonical_spoof_attack(token: str) -> str | None:
    if token == "f0" or token.startswith("f0_"):
        parts = token.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            return f"f0_{parts[1]}"
        return token
    if token.startswith("pitch_shift") or token.startswith("pitchshift"):
        return token
    if token.startswith("speed_change") or token.startswith("speedchange"):
        return token
    return None


def _label_from_parts(parts: Iterable[str]) -> tuple[str | None, str | None]:
    for part in parts:
        token = normalize_token(part)
        spoof_attack = _canonical_spoof_attack(token)
        if spoof_attack:
            return "spoof", spoof_attack
        for label, aliases in CLASS_ALIASES.items():
            if token in aliases:
                return label, token
        if "genuine" in token or "bona_fide" in token or "bonafide" in token:
            return "genuine", token
        if "spoof" in token or "synthetic" in token or "fake" in token:
            return "spoof", token
    return None, None


def collect_audio(root: Path) -> list[AudioItem]:
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"data root does not exist: {root}")

    items: list[AudioItem] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        label, attack_type = _label_from_parts(path.relative_to(root).parts)
        if label is None:
            continue
        items.append(AudioItem(path=path, label=label, attack_type=attack_type or label))
    return items


def _choose_train_test(pool: list[AudioItem], train_count: int, test_count: int, rng: random.Random) -> tuple[list[AudioItem], list[AudioItem]]:
    shuffled = list(pool)
    rng.shuffle(shuffled)

    requested = train_count + test_count
    if len(shuffled) >= requested:
        return shuffled[:train_count], shuffled[train_count:requested]

    if not shuffled:
        return [], []

    ratio = train_count / requested
    scaled_train = round(len(shuffled) * ratio)
    scaled_train = max(1, scaled_train) if train_count > 0 else 0
    scaled_train = min(scaled_train, len(shuffled))
    scaled_test = len(shuffled) - scaled_train

    if test_count > 0 and scaled_test == 0 and len(shuffled) > 1:
        scaled_train -= 1
        scaled_test = 1

    return shuffled[:scaled_train], shuffled[scaled_train:scaled_train + scaled_test]


def _allocate_counts(total: int, bucket_count: int) -> list[int]:
    base, remainder = divmod(total, bucket_count)
    return [base + (1 if idx < remainder else 0) for idx in range(bucket_count)]


def split_balanced(
    items: list[AudioItem],
    train_genuine: int,
    test_genuine: int,
    train_spoof: int,
    test_spoof: int,
    seed: int = 42,
) -> dict[str, list[AudioItem]]:
    genuine = [item for item in items if item.label == "genuine"]
    spoof = [item for item in items if item.label == "spoof"]

    if not genuine:
        raise ValueError("no genuine audio files were found")
    if not spoof:
        raise ValueError("no spoof audio files were found")

    rng = random.Random(seed)
    train_g, test_g = _choose_train_test(genuine, train_genuine, test_genuine, rng)
    train_s, test_s = _choose_train_test(spoof, train_spoof, test_spoof, rng)

    return {
        "train_genuine": train_g,
        "train_spoof": train_s,
        "test_genuine": test_g,
        "test_spoof": test_s,
    }


def split_balanced_by_spoof_attack(
    items: list[AudioItem],
    train_genuine: int,
    test_genuine: int,
    train_spoof: int,
    test_spoof: int,
    spoof_attacks: Iterable[str],
    seed: int = 42,
) -> dict[str, list[AudioItem]]:
    attacks = list(spoof_attacks)
    if not attacks:
        return split_balanced(items, train_genuine, test_genuine, train_spoof, test_spoof, seed)

    genuine = [item for item in items if item.label == "genuine"]
    if not genuine:
        raise ValueError("no genuine audio files were found")

    rng = random.Random(seed)
    train_g, test_g = _choose_train_test(genuine, train_genuine, test_genuine, rng)
    train_s: list[AudioItem] = []
    test_s: list[AudioItem] = []
    train_allocations = _allocate_counts(train_spoof, len(attacks))
    test_allocations = _allocate_counts(test_spoof, len(attacks))

    for attack, train_count, test_count in zip(attacks, train_allocations, test_allocations):
        pool = [item for item in items if item.label == "spoof" and item.attack_type == attack]
        if not pool:
            raise ValueError(f"no spoof audio files were found for attack/source: {attack}")
        attack_train, attack_test = _choose_train_test(pool, train_count, test_count, rng)
        train_s.extend(attack_train)
        test_s.extend(attack_test)

    rng.shuffle(train_s)
    rng.shuffle(test_s)
    return {
        "train_genuine": train_g,
        "train_spoof": train_s,
        "test_genuine": test_g,
        "test_spoof": test_s,
    }


def write_manifest(splits: dict[str, list[AudioItem]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "label", "attack_type", "path"])
        writer.writeheader()
        for split_name, rows in splits.items():
            for item in rows:
                writer.writerow(
                    {
                        "split": split_name,
                        "label": item.label,
                        "attack_type": item.attack_type,
                        "path": str(item.path),
                    }
                )


def read_manifest(path: Path) -> dict[str, list[AudioItem]]:
    splits: dict[str, list[AudioItem]] = {
        "train_genuine": [],
        "train_spoof": [],
        "test_genuine": [],
        "test_spoof": [],
    }
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            item = AudioItem(
                path=Path(row["path"]),
                label=row["label"],
                attack_type=row.get("attack_type") or row["label"],
            )
            splits.setdefault(row["split"], []).append(item)
    return splits


def summarize_splits(splits: dict[str, list[AudioItem]]) -> dict[str, int]:
    return {name: len(items) for name, items in splits.items()}
