# -*- coding: utf-8 -*-
"""
Keep the original tree, but inside each leaf create train/ and test/ folders.
For each (top, leaf):
  - Collect ALL audio files under that leaf (recursive)
  - Shuffle and split by ratio (default 50/50)
  - COPY files into:
        OUTPUT_ROOT / <top> / <leaf> / train | test / <original-filenames>
  - If files live directly under <top>, we treat leaf as "_root".

Safe: copies files (ไม่ย้าย). รองรับชื่อซ้ำด้วยการเติม suffix (_1, _2, ...).
เวอร์ชันนี้ "ไม่ลบไฟล์" ต้นฉบับ แต่จะข้ามไฟล์ขยะและไฟล์ AppleDouble (._*)
"""

import os
import csv
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple

# ===================== CONFIG =====================
INPUT_ROOT   = r"D:\NECTEC\Thai Spoof\Thaispoof_eq_split"          # โฟลเดอร์รากข้อมูลต้นทาง
OUTPUT_ROOT  = r"D:\NECTEC\Thai Spoof\Thaispoof_eq" # โฟลเดอร์ปลายทาง
AUDIO_EXTS   = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}  # นามสกุลเสียงที่ยอมรับ
TRAIN_RATIO  = 0.5                                        # สัดส่วน train (เช่น 0.8 = 80/20)
RANDOM_SEED  = 42
DRY_RUN      = False                                      # True = เขียนแต่ CSV/console ไม่ copy จริง
# ==================================================

# รายชื่อไฟล์ขยะยอดฮิต
SKIP_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}

def should_skip(p: Path) -> bool:
    """
    ข้ามไฟล์ขยะ/ซ่อน/ชั่วคราว และไฟล์ที่ไม่ใช่เสียง/ขนาด 0 byte
    - AppleDouble ของ macOS จะเป็นชื่อขึ้นต้นด้วย '._'
    - ไฟล์ชั่วคราว MS Office บางทีขึ้นต้นด้วย '~$'
    """
    n = p.name
    if n in SKIP_NAMES or n.startswith("._") or n.startswith("~$"):
        return True
    # ต้องเป็นไฟล์เสียงตามที่ระบุ และต้องมีขนาด > 0
    try:
        if p.suffix.lower() not in AUDIO_EXTS:
            return True
        if p.stat().st_size <= 0:
            return True
    except OSError:
        # กรณีไฟล์อ่าน stat ไม่ได้ ให้ข้าม
        return True
    return False

def is_audio(p: Path) -> bool:
    return p.suffix.lower() in AUDIO_EXTS

def leaf_of(file: Path, top_dir: Path) -> str:
    """leaf = โฟลเดอร์ชั้นแรกใต้ top; ถ้าไฟล์อยู่ใต้ top ตรง ๆ ให้ใช้ '_root'"""
    rel = file.relative_to(top_dir)
    parts = rel.parts
    return parts[0] if len(parts) > 1 else "_root"

def gather_files_by_leaf(input_root: Path) -> Tuple[Dict[Tuple[str, str], List[Path]], int]:
    """
    สร้าง mapping (top, leaf) -> รายการไฟล์เสียงทั้งหมดใต้ leaf นั้น (recursive)
    ข้ามไฟล์ขยะด้วย should_skip()
    return: (buckets, skipped_count)
    """
    buckets: Dict[Tuple[str, str], List[Path]] = {}
    skipped = 0

    for top_dir in sorted([p for p in input_root.iterdir() if p.is_dir()]):
        for f in top_dir.rglob("*"):
            if not f.is_file():
                continue
            if should_skip(f):
                skipped += 1
                continue
            key = (top_dir.name, leaf_of(f, top_dir))
            buckets.setdefault(key, []).append(f)

    return buckets, skipped

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def split_indices(n: int, ratio: float) -> Tuple[List[int], List[int]]:
    """สุ่ม index ทั้งหมดแล้วแบ่งเป็น train/test ตาม ratio"""
    idxs = list(range(n))
    random.shuffle(idxs)
    k = int(round(n * ratio))
    return idxs[:k], idxs[k:]

def non_clobber_path(dst_dir: Path, name: str) -> Path:
    """
    เลี่ยงการชนชื่อไฟล์: ถ้ามีชื่อซ้ำ จะเติม _1, _2, ...
    """
    base = Path(name).stem
    ext  = Path(name).suffix
    cand = dst_dir / (base + ext)
    i = 1
    while cand.exists():
        cand = dst_dir / f"{base}_{i}{ext}"
        i += 1
    return cand

def main():
    random.seed(RANDOM_SEED)

    in_root  = Path(INPUT_ROOT)
    out_root = Path(OUTPUT_ROOT)
    ensure_dir(out_root)

    if not in_root.exists():
        print(f"[ERROR] INPUT_ROOT not found: {in_root}")
        return

    buckets, skipped = gather_files_by_leaf(in_root)

    # CSVs
    manifest_rows = [("top", "leaf", "split", "dst_path", "src_path")]
    summary_rows  = [("top", "leaf", "total", "train", "test")]

    total_files = 0
    for (top, leaf), files in sorted(buckets.items()):
        files = sorted(files)
        n = len(files)
        if n == 0:
            continue

        train_idx, test_idx = split_indices(n, TRAIN_RATIO)

        # เตรียมโฟลเดอร์ปลายทาง
        train_dir = out_root / top / leaf / "train"
        test_dir  = out_root / top / leaf / "test"
        ensure_dir(train_dir)
        ensure_dir(test_dir)

        # คัดลอก train
        for idx in train_idx:
            src = files[idx]
            dst = non_clobber_path(train_dir, src.name)
            manifest_rows.append((top, leaf, "train", str(dst), str(src)))
            if not DRY_RUN:
                shutil.copy2(src, dst)

        # คัดลอก test
        for idx in test_idx:
            src = files[idx]
            dst = non_clobber_path(test_dir, src.name)
            manifest_rows.append((top, leaf, "test", str(dst), str(src)))
            if not DRY_RUN:
                shutil.copy2(src, dst)

        summary_rows.append((top, leaf, n, len(train_idx), len(test_idx)))
        total_files += n
        print(f"[OK] {top}/{leaf}: total={n} -> train={len(train_idx)}, test={len(test_idx)}")

    # เขียน CSV
    with open(out_root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(manifest_rows)

    with open(out_root / "summary_counts.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(summary_rows)

    print(f"\nDone. Total files processed: {total_files}")
    print(f"Skipped (not copied/indexed): {skipped} files (._*, .DS_Store, Thumbs.db, desktop.ini, non-audio, 0-byte, etc.)")
    print(f"Output root: {out_root}")
    if DRY_RUN:
        print("[NOTE] DRY_RUN=True -> ไม่มีการคัดลอกไฟล์จริง ๆ")

if __name__ == "__main__":
    main()
