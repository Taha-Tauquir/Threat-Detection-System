# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 20:24:02 2025

@author: Hp
"""
import json
import os

# === FILE PATHS ===
json_file = "C:/Users/Hp/Downloads/Dataset/2021_05_11_ph_cc001/images_0/annotations.json"
label_file = "C:/Users/Hp/Downloads/Dataset/2021_05_11_ph_cc001/labels_modified.txt"
output_file = "C:/Users/Hp/Downloads/Dataset/2021_05_11_ph_cc001/label_2.0.txt"

# === LOAD JSON ANNOTATIONS ===
with open(json_file, "r") as f:
    annotation_data = json.load(f)

# === PARSE JSON INTO FRAME ID DICT ===
annotation_dict = {}
for filename, info in annotation_data.items():
    try:
        frame_id = int(filename.split('.')[0].lstrip('0'))
        cx, cy = info["center"]
        w = info["width"]
        h = info["height"]
        annotation_dict[frame_id] = (cx, cy, w, h)
    except Exception as e:
        print(f"⚠️ Error parsing {filename}: {e}")

print(f"[✔] Loaded annotations for {len(annotation_dict)} frames")

# === PROCESS LABEL FILE ===
updated_lines = []
with open(label_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        try:
            frame_id = int(parts[0])
            if frame_id not in annotation_dict:
                continue
            cx, cy, w, h = annotation_dict[frame_id]
            updated_line = f"{frame_id} 1 {cx} {cy} {w} {h} {parts[6]} {parts[7]} {parts[8]}"
            updated_lines.append(updated_line)
        except Exception as e:
            print(f"❌ Error processing label line: {line} → {e}")

# === WRITE UPDATED LABEL FILE ===
with open(output_file, "w") as f:
    f.write("\n".join(updated_lines))

print(f"\n[✅] label_2.0.txt written with {len(updated_lines)} updated entries.")

