import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import csv

# ==== แก้ให้ตรงกับของคุณ ====
SOURCE = Path(r"D:\NECTEC\Thai Spoof\Thaispoof_eq_fixedcounts")
OUTPUT = Path(r"D:\NECTEC\Thai Spoof\Thaispoof_eq_split_SPLIT")

# ชื่อ split ที่ต้องการสร้าง
SPLITS = ["setA", "setB", "setC", "setD", "Test"]

# สุ่มให้เหมือนกันทุกครั้ง (ถ้าอยากเปลี่ยนก็เปลี่ยน seed ได้)
random.seed(42)

# เก็บสถิติเพื่อทำตาราง
# key = (dataset_label, degree_label, split_name)
stats = defaultdict(int)


def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def split_and_record(category_label: str, degree_label: str, src_folder: Path):
    """
    แบ่งไฟล์ในโฟลเดอร์ src_folder เป็น setA/setB/setC/setD/Test แบบสุ่ม
    และเก็บจำนวนไฟล์ลง stats
    """
    wav_files = [p for p in src_folder.rglob("*.wav")]
    if not wav_files:
        print(f"[WARN] ไม่พบ .wav ใน {src_folder}")
        return

    random.shuffle(wav_files)
    total = len(wav_files)
    print(f"[{category_label} | {degree_label}] พบไฟล์ทั้งหมด {total}")

    n_splits = len(SPLITS)
    base = total // n_splits
    remainder = total % n_splits

    # แบ่งให้แต่ละ split ต่างกันไม่เกิน 1 ไฟล์
    counts_per_split = []
    for i in range(n_splits):
        extra = 1 if i < remainder else 0
        counts_per_split.append(base + extra)

    idx = 0
    for split_name, split_count in zip(SPLITS, counts_per_split):
        split_files = wav_files[idx: idx + split_count]
        idx += split_count

        for wav_path in split_files:
            # โครงสร้างใหม่: OUTPUT / (path relative to SOURCE) / split / filename
            rel = wav_path.relative_to(SOURCE)
            dst = OUTPUT / rel.parent / split_name / wav_path.name
            safe_copy(wav_path, dst)

            # บันทึกสถิติ
            stats[(category_label, degree_label, split_name)] += 1

        print(f"  -> {split_name}: {len(split_files)} ไฟล์")


def main():
    # เดินดูทุกหมวดหมู่ใน ROOT
    for cat in SOURCE.iterdir():
        if not cat.is_dir():
            continue

        cat_name = cat.name

        print("\n========== PROCESSING:", cat_name, "==========")

        # กรณี F0 มี subfolder เป็น degree (เช่น F0_10, F0_40, ...)
        if cat_name == "F0":
            for degree_folder in cat.iterdir():
                if degree_folder.is_dir():
                    dataset_label = "F0 changing"  # ชื่อไว้ใช้ในตาราง
                    degree_label = degree_folder.name  # เช่น F0_10
                    split_and_record(dataset_label, degree_label, degree_folder)

        # หมวดอื่น ๆ ที่ไม่มี degree ย่อย (หรือยังไม่อยากแยก)
        else:
            if cat_name.lower() == "genuine":
                dataset_label = "Genuine"
                degree_label = "-"
            elif "tts" in cat_name.lower():
                dataset_label = "TTS"
                degree_label = "-"
            elif "pitch" in cat_name.lower():
                dataset_label = "Pitch shifting"
                degree_label = "-"
            else:
                # ถ้าไม่เข้ากลุ่มไหน ก็ใช้ชื่อโฟลเดอร์เป็น dataset
                dataset_label = cat_name
                degree_label = "-"

            split_and_record(dataset_label, degree_label, cat)

    # ===== สร้างตารางสรุปเป็น CSV =====
    summary_csv = OUTPUT / "split_summary_table.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    # เตรียม row ตาม Dataset + Degree
    rows = []
    grand_total = 0

    # list key คู่ (dataset, degree) ที่มีจริง
    pair_keys = sorted({(d, deg) for (d, deg, s) in stats.keys()})

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Degree"] + SPLITS + ["Total"])

        for dataset_label, degree_label in pair_keys:
            counts = []
            for split_name in SPLITS:
                cnt = stats.get((dataset_label, degree_label, split_name), 0)
                counts.append(cnt)
            row_total = sum(counts)
            grand_total += row_total

            writer.writerow([dataset_label, degree_label] + counts + [row_total])

        # แถวรวมทั้งหมด
        writer.writerow([])
        writer.writerow(["Total", "-", "", "", "", "", "", grand_total])

    print("\n=== เสร็จสิ้นการแบ่งไฟล์แล้ว ===")
    print("ผลลัพธ์ไฟล์อยู่ที่:", OUTPUT)
    print("ตารางสรุปเซฟไว้ที่:", summary_csv)


if __name__ == "__main__":
    main()
