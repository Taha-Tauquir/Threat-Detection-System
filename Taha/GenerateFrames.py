import os
import cv2
from pathlib import Path

def get_all_trimmed_video_paths(sessions_dir, extension=".avi"):
    video_paths = []
    for subdir, _, files in os.walk(sessions_dir):
        for file in files:
            if file.endswith(extension) and "_trimmed" in file:
                print(f"File Name is : {file}")
                video_paths.append(os.path.join(subdir, file))
    return video_paths

def extract_frames(video_path, output_base_dir, target_fps=30, target_dir="train"):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    session_id = os.path.basename(os.path.dirname(video_path))
    output_dir = os.path.join(output_base_dir, target_dir, session_id, f"{video_name}_frames")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_interval = 1.0 / target_fps
    next_save_time = 0.0 
    saved_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 

        if current_time >= next_save_time:
            time_diff_ms = (current_time - next_save_time) * 1000
            print(f"Saving frame {saved_idx} at time: {current_time * 1000:.0f} ms {time_diff_ms:.2f} ms")
            
            cv2.imwrite(os.path.join(output_dir, f"{str(saved_idx).zfill(4)}.png"), frame)
            saved_idx += 1
            next_save_time += frame_interval

    cap.release()
    print(f"Saved {saved_idx - 1} frames from {video_name} at {target_fps} FPS in: {output_dir}")



def iterate_through_videos(root_dir, target_dir):
    dataset_dir = Path(root_dir)
    output_dir = dataset_dir.parent
    video_paths = get_all_trimmed_video_paths(dataset_dir)

    if video_paths:
        for video_path in video_paths:
            extract_frames(
                video_path=video_path,
                output_base_dir=output_dir,
                target_fps=30,
                target_dir=target_dir
            )
    else:
        print("No '_trimmed.avi' video files found in folder.")


# if __name__ == "__main__":
#     iterate_through_videos()