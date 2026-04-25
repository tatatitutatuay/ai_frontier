#!/usr/bin/env python3
"""
split_thaispoof.py

Usage: ปรับตัวแปร SOURCE / OUTPUT ด้านล่างให้ตรงกับเครื่องคุณ แล้วรัน:
    python split_thaispoof.py

ผลลัพธ์จะอยู่ที่ OUTPUT (โฟลเดอร์ train/ test) และมีไฟล์ split_summary.csv
"""

import random
import shutil
import csv
import math
from pathlib import Path

# ================== ปรับค่าตรงนี้ ==================
SOURCE = Path(r"D:\NECTEC\Thai Spoof\Thaispoof_eq_split")        # โฟลเดอร์ต้นทาง
OUTPUT = Path(r"D:\NECTEC\Thai Spoof\Thaispoof_eq_split_final")  # โฟลเดอร์ผลลัพธ์

# เป้าหมายเดิมที่คุยกัน
DESIRED_TRAIN_G = 3000
DESIRED_TEST_G  = 1500
DESIRED_TRAIN_S = 3000
DESIRED_TEST_S  = 1500

# การตั้งค่าอื่น ๆ
RANDOM_SEED = 42     # ถ้าต้องการผลซ้ำ ให้เก็บค่า; ตั้ง None เพื่อสุ่มจริง ๆ
COPY_INSTEAD_OF_MOVE = True  # True = copy files, False = move files (เปลี่ยนตามต้องการ)
VERBOSE = True
# ====================================================

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

def log(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

# --- helper: หาไฟล์ genuine แบบ recursive ---
def find_genuine_files(root: Path):
    found = []
    if not root.exists():
        return []
    # หาโฟลเดอร์ที่ชื่อมี 'genuine' (case-insensitive) แล้วรวมไฟล์ .wav ภายใน (ทุกชั้น)
    for d in root.rglob("*"):
        if d.is_dir() and "genuine" in d.name.lower():
            found.extend([p for p in d.rglob("*") if p.suffix.lower() == ".wav"])
    # fallback: ถ้าไม่เจอ ให้ดูไฟล์ wav ที่อยู่ตรง ๆ ใต้ SOURCE
    if not found:
        found = [p for p in root.glob("*.wav") if p.suffix.lower() == ".wav"]
    # dedupe and sort
    unique = sorted({p.resolve() for p in found})
    return unique

# --- helper: หา spoof leaf folders (exclude any folder that has 'genuine' in name) ---
def find_spoof_leaf_folders(root: Path, exclude_name_keyword="genuine"):
    leaves = []
    if not root.exists():
        return []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if exclude_name_keyword.lower() in child.name.lower():
            continue
        # If this folder itself contains wavs (directly or in subfolders), find the actual leaf folders that contain wavs
        # We consider leaf folders as the deepest folders that contain wav files.
        # Walk all subfolders and collect those that directly contain wavs.
        for sub in child.rglob("*"):
            if sub.is_dir():
                if any(p.suffix.lower() == ".wav" for p in sub.iterdir()):
                    leaves.append(sub)
        # also if child folder directly contains wavs, include it
        if any(p.suffix.lower() == ".wav" for p in child.iterdir()):
            leaves.append(child)
    # dedupe keeping stable order
    unique = []
    seen = set()
    for p in leaves:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique

# --- prepare output folders ---
def prepare_output(out: Path):
    (out / "train" / "genuine").mkdir(parents=True, exist_ok=True)
    (out / "test"  / "genuine").mkdir(parents=True, exist_ok=True)
    (out / "train" / "spoof").mkdir(parents=True, exist_ok=True)
    (out / "test"  / "spoof").mkdir(parents=True, exist_ok=True)

# --- copy or move wrapper ---
def copy_or_move(src: Path, dst: Path):
    if COPY_INSTEAD_OF_MOVE:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))

# ---------------- main ----------------
def main():
    log("SOURCE:", SOURCE)
    log("OUTPUT:", OUTPUT)
    if not SOURCE.exists():
        log("Error: SOURCE path does not exist.")
        return

    prepare_output(OUTPUT)

    # find genuine
    genuine_files = find_genuine_files(SOURCE)
    log(f"Genuine files detected: {len(genuine_files)}")
    if len(genuine_files) > 0:
        for f in genuine_files[:10]:
            log("  sample genuine:", f.relative_to(SOURCE))
    else:
        log("  (no genuine files found)")

    # find spoof types
    spoof_types = find_spoof_leaf_folders(SOURCE)
    log(f"Detected spoof types (leaf folders): {len(spoof_types)}")
    for t in spoof_types:
        log("  -", t.relative_to(SOURCE))

    # ========== decide genuine split ==========
    total_desired_g = DESIRED_TRAIN_G + DESIRED_TEST_G
    avail_g = len(genuine_files)
    if avail_g >= total_desired_g:
        use_train_g = DESIRED_TRAIN_G
        use_test_g  = DESIRED_TEST_G
        log(f"Using desired genuine counts: Train={use_train_g}, Test={use_test_g}")
    else:
        if avail_g == 0:
            log("WARNING: ไม่พบไฟล์ genuine เลยใน SOURCE. genuine จะเป็น 0.")
            use_train_g = 0
            use_test_g  = 0
        else:
            log(f"NOTE: genuine ไม่พอ (มี {avail_g} ต้องการ {total_desired_g}). จะใช้ไฟล์ทั้งหมดแล้วแบ่งเป็น 2:1 (train:test).")
            use_train_g = math.floor(avail_g * 2 / 3)
            use_test_g  = avail_g - use_train_g
            log(f"Adjusted genuine split -> Train={use_train_g}, Test={use_test_g}")

    # perform genuine split & copy/move
    train_g_files = []
    test_g_files = []
    if avail_g > 0 and (use_train_g + use_test_g > 0):
        random.shuffle(genuine_files)
        train_g_files = genuine_files[:use_train_g]
        test_g_files  = genuine_files[use_train_g:use_train_g + use_test_g]
        for f in train_g_files:
            dst = OUTPUT / "train" / "genuine" / f.name
            copy_or_move(f, dst)
        for f in test_g_files:
            dst = OUTPUT / "test"  / "genuine" / f.name
            copy_or_move(f, dst)

    log(f"Genuine copied/moved: Train={len(train_g_files)}, Test={len(test_g_files)}, Available={avail_g}")

    # ========== split spoof equally ==========
    if not spoof_types:
        log("Error: ไม่พบ spoof types ที่มีไฟล์ .wav -- กรุณาตรวจโครงสร้างโฟลเดอร์")
        return

    num_types = len(spoof_types)
    base_train_per = DESIRED_TRAIN_S // num_types
    base_test_per  = DESIRED_TEST_S  // num_types
    rem_train = DESIRED_TRAIN_S - base_train_per * num_types
    rem_test  = DESIRED_TEST_S  - base_test_per * num_types

    # distribute remainders to first few types
    train_per_type = [base_train_per + (1 if i < rem_train else 0) for i in range(num_types)]
    test_per_type  = [base_test_per  + (1 if i < rem_test  else 0) for i in range(num_types)]

    summary_rows = []

    for idx, tpath in enumerate(spoof_types):
        # collect wavs in this type (recursive inside the leaf folder)
        all_wavs = [p for p in tpath.rglob("*") if p.suffix.lower() == ".wav"]
        all_wavs = sorted({p.resolve() for p in all_wavs})
        need_train = train_per_type[idx]
        need_test  = test_per_type[idx]

        log(f"Type '{tpath.relative_to(SOURCE)}' has {len(all_wavs)} available. Need Train={need_train}, Test={need_test}")

        if len(all_wavs) < need_train + need_test:
            # Not enough files in this type => stop and inform
            msg = (f"ERROR: ไฟล์ไม่พอใน type '{tpath.relative_to(SOURCE)}' : "
                   f"มี {len(all_wavs)} ต้องการ {need_train + need_test}. ปรับจำนวนหรือเติมไฟล์ก่อนรันใหม่.")
            log(msg)
            raise SystemExit(msg)

        random.shuffle(all_wavs)
        chosen_train = all_wavs[:need_train]
        chosen_test  = all_wavs[need_train:need_train + need_test]

        # make a safe type_name for output folder (join relative path parts with underscore)
        type_name = "_".join(tpath.relative_to(SOURCE).parts)
        out_train = OUTPUT / "train" / "spoof" / type_name
        out_test  = OUTPUT / "test"  / "spoof" / type_name
        out_train.mkdir(parents=True, exist_ok=True)
        out_test.mkdir(parents=True, exist_ok=True)

        for f in chosen_train:
            dst = out_train / f.name
            copy_or_move(f, dst)
        for f in chosen_test:
            dst = out_test / f.name
            copy_or_move(f, dst)

        log(f" -> {type_name}: Train={len(chosen_train)}, Test={len(chosen_test)}, Available={len(all_wavs)}")
        summary_rows.append({
            "Dataset": "spoof",
            "Degree": type_name,
            "Train": len(chosen_train),
            "Test": len(chosen_test),
            "TotalAvailable": len(all_wavs),
            "TotalSelected": len(chosen_train) + len(chosen_test)
        })

    # add genuine summary row
    summary_rows.append({
        "Dataset": "genuine",
        "Degree": "genuine",
        "Train": len(train_g_files),
        "Test": len(test_g_files),
        "TotalAvailable": avail_g,
        "TotalSelected": len(train_g_files) + len(test_g_files)
    })

    # save CSV summary
    csv_path = OUTPUT / "split_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=["Dataset","Degree","Train","Test","TotalAvailable","TotalSelected"])
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    log("=== เสร็จสิ้น ===")
    log("Summary saved to:", csv_path)
    log("Output folder structure sample:")
    log(" -", OUTPUT / "train")
    log(" -", OUTPUT / "test")

if __name__ == "__main__":
    main()
