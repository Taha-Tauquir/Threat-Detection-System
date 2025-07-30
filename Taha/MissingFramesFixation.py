import os
import cv2
import numpy as np

def load_frame(path):
    return cv2.imread(path).astype(np.float32)

def save_frame(img, path):
    cv2.imwrite(path, img.astype(np.uint8))

def interpolate_frames(prev_path, next_path, missing_indices, save_dir):
    prev_img = load_frame(prev_path)
    next_img = load_frame(next_path)
    total = len(missing_indices) + 1

    for i, frame_idx in enumerate(missing_indices, 1):
        alpha = i / (total)
        interpolated = cv2.addWeighted(prev_img, 1 - alpha, next_img, alpha, 0)
        save_path = os.path.join(save_dir, f"{str(frame_idx).zfill(4)}.png")
        save_frame(interpolated, save_path)
        print(f"Interpolated frame: {interpolated}")
        print(f"Interpolated path: {save_path}")

def duplicate_frame(source_path, target_indices, save_dir):
    img = load_frame(source_path)
    for idx in target_indices:
        save_path = os.path.join(save_dir, f"{str(idx).zfill(4)}.png")
        save_frame(img, save_path)
       
        print(f"Duplicated {source_path} to {save_path}")

def find_existing_frame_indices(folder):
    return sorted([
        int(f.split(".")[0])
        for f in os.listdir(folder)
        if f.endswith(".png") and f != "inconsistent_frames_folder"
    ])

def fill_missing_frames(folder_path, total_frames=900, folder_ends='some_folder'):
    # Step 1: Find face-detected folder (ending with _frames)
    frames_subfolder = None
    for name in os.listdir(folder_path):
        full = os.path.join(folder_path, name)
        if os.path.isdir(full) and name.endswith(folder_ends):
            frames_subfolder = full
            break

    if frames_subfolder is None:
        print(f"No '_frames' folder found in {folder_path}")
        return
    else:
        print(f"Using frames from: {frames_subfolder}")

    # Step 2: Check for inconsistent frames
    inconsistent_dir = os.path.join(folder_path, "inconsistent_frames_folder")
    if not os.path.exists(inconsistent_dir):
        print("No inconsistent_frames_folder folder found.")
        return

    missing_frames = sorted([
        int(f.split(".")[0]) for f in os.listdir(inconsistent_dir)
        if f.endswith(".png")
    ])
    if not missing_frames:
        print("No missing frames to process.")
        return

    existing_frames = sorted([
        int(f.split(".")[0])
        for f in os.listdir(frames_subfolder)
        if f.endswith(".png") and not f.startswith("._")
    ])

    if not existing_frames:
        print("No existing frames found in the frames folder.")
        return

    full_set = set(range(1, total_frames + 1))
    missing_set = sorted(list(full_set - set(existing_frames)))
    groups = group_consecutive(missing_set)

    for group in groups:
        start = group[0]
        end = group[-1]

        prev_idx = start - 1 if start > 1 else None
        next_idx = end + 1 if end < total_frames else None

        prev_path = os.path.join(frames_subfolder, f"{str(prev_idx).zfill(4)}.png") if prev_idx else None
        next_path = os.path.join(frames_subfolder, f"{str(next_idx).zfill(4)}.png") if next_idx else None

        if prev_path and next_path and os.path.exists(prev_path) and os.path.exists(next_path):
            interpolate_frames(prev_path, next_path, group, frames_subfolder)
        elif prev_path and os.path.exists(prev_path):
            duplicate_frame(prev_path, group, frames_subfolder)
        elif next_path and os.path.exists(next_path):
            duplicate_frame(next_path, group, frames_subfolder)
        else:
            print(f"Cannot fix group {group}: No valid neighboring frames found.")

def group_consecutive(nums):
    if not nums:
        return []
    nums = sorted(nums)
    groups = []
    group = [nums[0]]

    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1] + 1:
            group.append(nums[i])
        else:
            groups.append(group)
            group = [nums[i]]
    groups.append(group)
    return groups

def FixMissingFrames(base_dir):
    for folder in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, folder)
        fill_missing_frames(full_path, total_frames=900, folder_ends="_frames")

# if __name__ == "__main__":
#     base_dir = '/Users/mzeeshan/Documents/PythonProjects/NewRppgHundred/train'
#     FixMissingFrames(base_dir)
