# -*- coding: utf-8 -*-
"""
Count audio files in each subfolder (recursive) under 6 target folders.

Target Folders (under ROOT):
  F0, genuine, MMS_spoof, PitchShift, SpeedChange, TTS_VAJA_Spoof

Output:
  folder_counts_detailed.csv
    Columns: top_folder, sub_folder, file_count
"""

import os
import csv
from pathlib import Path

# ---------- CONFIG ----------
ROOT = r"D:\NECTEC\Thai Spoof\Thaispoof"
FOLDERS = ["F0", "genuine", "MMS_spoof", "PitchShift", "SpeedChange", "TTS_VAJA_Spoof"]
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
OUTPUT_CSV = str(Path(ROOT) / "folder_counts_detailed.csv")
# ----------------------------

def count_files(folder_path: Path, exts) -> int:
    count = 0
    if not folder_path.exists():
        return 0
    for root, _, files in os.walk(folder_path):
        for f in files:
            if Path(f).suffix.lower() in exts:
                count += 1
    return count

def main():
    rows = []
    for top in FOLDERS:
        top_path = Path(ROOT) / top
        if not top_path.exists():
            rows.append({"top_folder": top, "sub_folder": "(missing)", "file_count": 0})
            continue

        # loop ทุก subfolder ถ้ามี ถ้าไม่มี subfolder ให้นับตรง top
        subfolders = [p for p in top_path.iterdir() if p.is_dir()]
        if not subfolders:
            n = count_files(top_path, AUDIO_EXTS)
            rows.append({"top_folder": top, "sub_folder": "(self)", "file_count": n})
        else:
            for sub in subfolders:
                n = count_files(sub, AUDIO_EXTS)
                rows.append({"top_folder": top, "sub_folder": sub.name, "file_count": n})

    # เขียน CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["top_folder", "sub_folder", "file_count"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved -> {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
