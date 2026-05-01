from __future__ import annotations

import csv
from collections import defaultdict
import random
import re
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


def _is_metadata_path(path: Path) -> bool:
    parts = {normalize_token(part) for part in path.parts}
    return "__macosx" in parts or any(part.startswith("._") for part in path.parts)


def _canonical_spoof_attack(token: str) -> str | None:
    f0_match = re.search(r"(?:^|_)f0_(\d+)(?:_|$)", token)
    if f0_match:
        return f"f0_{f0_match.group(1)}"
    if token.startswith("pitch_shift") or token.startswith("pitchshift"):
        return token
    if token.startswith("speed_change") or token.startswith("speedchange"):
        return token
    return None


def _label_from_parts(parts: Iterable[str]) -> tuple[str | None, str | None]:
    normalized = [normalize_token(part) for part in parts]
    for token in normalized:
        spoof_attack = _canonical_spoof_attack(token)
        if spoof_attack:
            return "spoof", spoof_attack

    for token in normalized:
        for label, aliases in CLASS_ALIASES.items():
            if token in aliases:
                return label, token
        if "genuine" in token or "bona_fide" in token or "bonafide" in token:
            return "genuine", "genuine"
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
        if _is_metadata_path(path):
            continue
        label, attack_type = _label_from_parts(path.relative_to(root).parts)
        if label is None:
            continue
        items.append(AudioItem(path=path, label=label, attack_type=attack_type or label))
    return items


def utterance_group_id(path: Path) -> str:
    """Return the base utterance ID shared by genuine and spoof variants."""
    stem = normalize_token(Path(path).stem)
    stem = re.sub(r"^f0_\d+_", "", stem)

    thai_match = re.search(r"thai_?0*(\d+)", stem)
    if thai_match:
        return f"thai{int(thai_match.group(1)):04d}"

    collapsed = re.sub(r"[^a-z0-9]+", "", stem)
    return collapsed or stem


def _group_key(item: AudioItem) -> str:
    return utterance_group_id(item.path)


def _group_category(rows: list[AudioItem]) -> tuple[str, ...]:
    labels = {item.label for item in rows}
    attacks = sorted({item.attack_type for item in rows if item.label == "spoof"})
    if labels == {"genuine"}:
        return ("genuine",)
    if labels == {"spoof"}:
        return ("spoof", *attacks)
    return ("mixed", *attacks)


def _split_keys_by_ratio(keys: list[str], ratio: float, rng: random.Random) -> tuple[set[str], set[str]]:
    shuffled = list(keys)
    rng.shuffle(shuffled)
    if not shuffled:
        return set(), set()

    train_count = round(len(shuffled) * ratio)
    if ratio > 0 and train_count == 0:
        train_count = 1
    if ratio < 1 and train_count == len(shuffled) and len(shuffled) > 1:
        train_count -= 1

    return set(shuffled[:train_count]), set(shuffled[train_count:])


def _split_group_keys(
    items: list[AudioItem],
    train_genuine: int,
    test_genuine: int,
    train_spoof: int,
    test_spoof: int,
    rng: random.Random,
) -> tuple[set[str], set[str]]:
    grouped: dict[str, list[AudioItem]] = defaultdict(list)
    for item in items:
        grouped[_group_key(item)].append(item)

    categories: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for key, rows in grouped.items():
        categories[_group_category(rows)].append(key)

    genuine_ratio = train_genuine / (train_genuine + test_genuine)
    spoof_ratio = train_spoof / (train_spoof + test_spoof)
    overall_ratio = (train_genuine + train_spoof) / (
        train_genuine + test_genuine + train_spoof + test_spoof
    )

    train_keys: set[str] = set()
    test_keys: set[str] = set()
    for category, keys in categories.items():
        if category[0] == "genuine":
            ratio = genuine_ratio
        elif category[0] == "spoof":
            ratio = spoof_ratio
        else:
            ratio = overall_ratio
        category_train, category_test = _split_keys_by_ratio(keys, ratio, rng)
        train_keys.update(category_train)
        test_keys.update(category_test)

    return train_keys, test_keys


def _take_from_keys(pool: list[AudioItem], keys: set[str], count: int, rng: random.Random) -> list[AudioItem]:
    candidates = [item for item in pool if _group_key(item) in keys]
    rng.shuffle(candidates)
    return candidates[: min(count, len(candidates))]


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
    train_keys, test_keys = _split_group_keys(
        items,
        train_genuine=train_genuine,
        test_genuine=test_genuine,
        train_spoof=train_spoof,
        test_spoof=test_spoof,
        rng=rng,
    )
    train_g = _take_from_keys(genuine, train_keys, train_genuine, rng)
    test_g = _take_from_keys(genuine, test_keys, test_genuine, rng)
    train_s = _take_from_keys(spoof, train_keys, train_spoof, rng)
    test_s = _take_from_keys(spoof, test_keys, test_spoof, rng)

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
    train_keys, test_keys = _split_group_keys(
        items,
        train_genuine=train_genuine,
        test_genuine=test_genuine,
        train_spoof=train_spoof,
        test_spoof=test_spoof,
        rng=rng,
    )
    train_g = _take_from_keys(genuine, train_keys, train_genuine, rng)
    test_g = _take_from_keys(genuine, test_keys, test_genuine, rng)
    train_s: list[AudioItem] = []
    test_s: list[AudioItem] = []
    train_allocations = _allocate_counts(train_spoof, len(attacks))
    test_allocations = _allocate_counts(test_spoof, len(attacks))

    for attack, train_count, test_count in zip(attacks, train_allocations, test_allocations):
        pool = [item for item in items if item.label == "spoof" and item.attack_type == attack]
        if not pool:
            raise ValueError(f"no spoof audio files were found for attack/source: {attack}")
        attack_train = _take_from_keys(pool, train_keys, train_count, rng)
        attack_test = _take_from_keys(pool, test_keys, test_count, rng)
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
