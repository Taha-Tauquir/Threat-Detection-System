import os
import torch
from torch.utils.data import Dataset
import numpy as np
from RadarPreprocessing import RadarPreprocessing as RadarPreprocessor

class RadarBBoxDataset(Dataset):
    def __init__(self, root_dir, cube_shape=(32, 32, 16)):
        self.samples = []
        self.preprocessor = RadarPreprocessor(cube_shape)
        self.img_w, self.img_h = 1440, 1080

        weapon_count = 0
        non_weapon_count = 0
        total_lines = 0
        loaded_samples = 0

        for session in os.listdir(root_dir):
            session_path = os.path.join(root_dir, session)
            radar_dir = os.path.join(session_path, "radar_raw_frame")
            label_file = os.path.join(session_path, "label_2.0.txt")

            if not (os.path.exists(label_file) and os.path.isdir(radar_dir)):
                continue

            with open(label_file, "r") as f:
                for line in f:
                    total_lines += 1
                    parts = line.strip().split()

                    if len(parts) < 8:
                        print(f" Skipped malformed line in {label_file}: {line.strip()}")
                        continue

                    try:
                        frame_id = int(parts[0])
                        x1, y1, x2, y2 = map(float, parts[2:6])
                        carry_type = parts[6].lower()
                        object_type = parts[7].lower()
                        weapon_label = 1.0 if "knife" in object_type else 0.0

                        if weapon_label == 1.0:
                            weapon_count += 1
                        else:
                            non_weapon_count += 1

                        # Normalize bbox
                        # x1, x2 = sorted([x1, x2])
                        # y1, y2 = sorted([y1, y2])
                        # cx = (x1 + x2) / 2 / self.img_w
                        # cy = (y1 + y2) / 2 / self.img_h
                        # w = (x2 - x1) / self.img_w
                        # h = (y2 - y1) / self.img_h
                        #min max normalization
                        cx = x1 / self.img_w
                        cy = y1 /  self.img_h
                        w = x2 / self.img_w
                        h = y2 /  self.img_h


                        radar_file = os.path.join(radar_dir, f"{frame_id:06d}.mat")

                        if not all(0 <= v <= 1 for v in [cx, cy, w, h]):
                            print(f"🚨 Abnormal BBox in {radar_file}: cx={cx:.4f}, cy={cy:.4f}, w={w:.4f}, h={h:.4f}")

                        if os.path.exists(radar_file):
                            self.samples.append((radar_file, [cx, cy, w, h], weapon_label))
                            loaded_samples += 1
                        else:
                            print(f" Missing radar file: {radar_file}")
                    except Exception as e:
                        print(f" Error in {label_file} line: {line.strip()} -> {e}")

        print(f"\n Dataset loading summary:")
        print(f"   Total lines parsed:     {total_lines}")
        print(f"   Valid samples loaded:   {loaded_samples}")
        print(f"   Weapon labels:          {weapon_count}")
        print(f"   Non-weapon labels:      {non_weapon_count}")
        print(f"   Total dataset size:     {len(self.samples)}\n")
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        radar_path, bbox, weapon_label = self.samples[idx]
        adc_data = self.preprocessor.preprocess(radar_path)
        if adc_data is None:
            print(f" Warning: Failed to preprocess {radar_path}, using zeros.")
            adc_data = np.zeros((32, 32, 16), dtype=np.float32)

        return (
            torch.tensor(adc_data, dtype=torch.float32).unsqueeze(0),
            torch.tensor(bbox, dtype=torch.float32),
            torch.tensor(weapon_label, dtype=torch.float32)  # scalar, not shape [1]
        )
