import os
import numpy as np
import cv2
from natsort import natsorted

import os
import numpy as np
import cv2
from natsort import natsorted


import os
import numpy as np
import cv2
from natsort import natsorted

def process_folder_to_multiple_chunks(
    frame_folder, output_dir, chunk_size=180, target_size=(72, 72), prefix="Input"
):
    os.makedirs(output_dir, exist_ok=True)
    frame_files = natsorted([f for f in os.listdir(frame_folder) if f.endswith(".png")])
    num_frames = len(frame_files)

    if num_frames < chunk_size:
        print(f"❌ Skipping: {frame_folder} (only {num_frames} frames)")
        return

    print(f"📁 Processing: {frame_folder} ({num_frames} frames)")

    # Load and preprocess all RGB frames
    frames_rgb = []
    for f in frame_files:
        img = cv2.imread(os.path.join(frame_folder, f))
        img = cv2.resize(img, target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames_rgb.append(img.astype(np.float32) / 255.0)

    frames_rgb = np.stack(frames_rgb)  # (T, H, W, 3)

    # Split into multiple non-overlapping chunks
    num_chunks = num_frames // chunk_size

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = start + chunk_size
        rgb = frames_rgb[start:end]  # (chunk_size, H, W, 3)

        # Compute frame differences
        diff = np.zeros_like(rgb)
        diff[1:] = rgb[1:] - rgb[:-1]

        # Transpose both RGB and diff to (T, C, H, W)
        rgb = rgb.transpose(0, 3, 1, 2)
        diff = diff.transpose(0, 3, 1, 2)

        combined = np.concatenate([rgb, diff], axis=1)  # (T, 6, H, W)

        output_path = os.path.join(output_dir, f"{prefix}_{chunk_idx+1:04d}.npy")
        np.save(output_path, combined)
        print(f"✅ Saved: {output_path} | Shape: {combined.shape}")


if __name__ == "__main__":
    frame_folder = "C:/Users/Hp/Downloads/MAHNOB-SAMPLE_DATASET/tst/428/P4-Rec1-2009.07.21.17.46.06_C1 trigger _C_Section_38_trimmed_frames"
    output_path = "C:/Users/Hp/Downloads/MAHNOB-SAMPLE_DATASET/tst/428/input"
    process_folder_to_multiple_chunks(frame_folder, output_path)
