import os
import cv2
import random
import shutil

# 1. Find all _trimmed.avi files from nested folders
def find_all_avi_files(root_dir):
    avi_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith('_trimmed.avi'):
                avi_files.append(os.path.join(root, f))
    return avi_files

# 2. Apply brightness/contrast transformation
def transform_video(input_path, output_path, brightness=0, contrast=0):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Cannot open: {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        modified = cv2.convertScaleAbs(frame, alpha=1 + contrast, beta=brightness)
        out.write(modified)

    cap.release()
    out.release()
    return True

# 3. Get new output path with suffix
def get_modified_path(original_path, suffix):
    dir_path = os.path.dirname(original_path)
    filename = os.path.splitext(os.path.basename(original_path))[0]
    return os.path.join(dir_path, f"{filename}_{suffix}.avi")

# 4. Main pipeline
# def process_trimmed_videos(input_root):
#     all_avi_files = find_all_avi_files(input_root)
#     random.shuffle(all_avi_files)
#     total = len(all_avi_files)
#     split = round(total / 3)
#
#     bright_files = all_avi_files[:split]
#     dark_files = all_avi_files[split:2*split]
#     original_files = all_avi_files[2*split:]
#
#     print(f"Total Videos: {total}")
#     print(f"Bright: {len(bright_files)}")
#     print(f"Dark: {len(dark_files)}")
#     print(f"Original: {len(original_files)}\n")
#
#     for f in bright_files:
#         out_path = get_modified_path(f, 'bright')
#         if transform_video(f, out_path, brightness=40, contrast=0.2):
#             os.remove(f)
#             print(f"Bright version saved: {out_path}")
#
#     for f in dark_files:
#         out_path = get_modified_path(f, 'dark')
#         if transform_video(f, out_path, brightness=-40, contrast=-0.2):
#             os.remove(f)
#             print(f"Dark version saved: {out_path}")
#
#     for f in original_files:
#         out_path = get_modified_path(f, 'original')
#         shutil.copyfile(f, out_path)
#         os.remove(f)
#         print(f"Original copy saved: {out_path}")
#
#     print("\nAll videos processed.")

# if __name__ == "__main__":
#     process_trimmed_videos()
