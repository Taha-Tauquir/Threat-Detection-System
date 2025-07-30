import os
import cv2
import numpy as np

def find_frames_folder(participant_path):
    for item in os.listdir(participant_path):
        if "_frames" in item.lower() and os.path.isdir(os.path.join(participant_path, item)):
            return os.path.join(participant_path, item)
    return None

def load_frames_from_folder(folder_path):
    image_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith('.png')],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    frames = []
    for file in image_files:
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (128, 128))
        frames.append(img)
    if len(frames) != 900:
        raise ValueError(f"Expected 900 frames in {folder_path}, but got {len(frames)}")
    return np.stack(frames)


def diff_normalized(frames_np):
    diff = frames_np[1:] - frames_np[:-1]
    norm = (diff - np.mean(diff)) / np.std(diff)

    zero_frame = np.zeros_like(norm[0:1])
    norm_padded = np.concatenate([zero_frame, norm], axis=0)  # Shape: (900, 128, 128, 3)

    return norm_padded.astype(np.float32)

# def diff_normalized(frames_np):
#     diff = frames_np[1:] - frames_np[:-1]
#     norm = (diff - np.mean(diff)) / np.std(diff)
#     return norm.astype(np.float32)

def process_inputs_npys(train_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    participant_ids = sorted(os.listdir(train_dir))

    for pid in participant_ids:
        participant_path = os.path.join(train_dir, pid)
        if not os.path.isdir(participant_path):
            continue

        frames_path = find_frames_folder(participant_path)
        if frames_path is None:
            print(f"No '_frames' folder found in {pid}, Skip!")
            continue

        try:
            print(f"Processing participant {pid}...")
            frames = load_frames_from_folder(frames_path)
            diffnorm = diff_normalized(frames)

            save_path = os.path.join(output_dir, f"{str(pid).zfill(4)}_input.npy")
            np.save(save_path, diffnorm)
            print(f"Saved: {save_path} | shape={diffnorm.shape}")

        except Exception as e:
            print(f"[ERROR] {pid}: {e}")

# if __name__ == "__main__":
#     train_directory = "/Users/mzeeshan/Documents/PythonProjects/NewRppg/train"
#     output_directory = "/Users/mzeeshan/Documents/PythonProjects/NewRppg/cached"
#     process_inputs_npys(train_directory, output_directory)
