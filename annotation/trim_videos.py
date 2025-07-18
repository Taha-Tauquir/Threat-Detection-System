
# import os
# from moviepy.editor import VideoFileClip

# root_dataset_dir = "/Users/mzeeshan/Documents/PythonProjects/FYP/Sessions"
# replace_original = True

# def trim_to_30_seconds(video_path):
#     try:
#         clip = VideoFileClip(video_path)
#         duration = clip.duration

#         if duration < 30:
#             print(f"Video skipped (shorter than 30sec): {video_path}")
#             return

#         start_time = (duration / 2) - 15
#         end_time = (duration / 2) + 15

#         temp_output = video_path + ".temp_trimmed.avi"

#         trimmed_clip = clip.subclip(start_time, end_time)
#         trimmed_clip.write_videofile(temp_output, codec="libx264", audio_codec="aac", verbose=False, logger=None)

#         if replace_original:
#             os.replace(temp_output, video_path)
#             print(f"Trimmed and replaced: {video_path}")
#         else:
#             print(f"Trimmed saved at: {temp_output}")
#     except Exception as e:
#         print(f"Error processing {video_path}: {e}")

# def main():
#     for root, dirs, files in os.walk(root_dataset_dir):
#         for file in files:
#             if file.lower().endswith(".avi"):
#                 video_path = os.path.join(root, file)
#                 trim_to_30_seconds(video_path)



import cv2
import os

def extract_center_clip(input_path, output_path, clip_duration_sec=30):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Cannot open video file: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    duration_sec = total_frames / fps
    if clip_duration_sec > duration_sec:
        print(f"Skipping {input_path}: video shorter than {clip_duration_sec} seconds.")
        cap.release()
        return

    start_time = (duration_sec - clip_duration_sec) / 2
    start_frame = int(start_time * fps)
    end_frame = start_frame + int(clip_duration_sec * fps)

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1

    cap.release()
    out.release()
    print(f"Saved trimmed video: {output_path}")

def process_all_videos_in_dataset(train_root_dir):
    for root, dirs, files in os.walk(train_root_dir):
        for file in files:
            if file.endswith('.avi') and '_trimmed' not in file:
                full_path = os.path.join(root, file)
                filename_wo_ext = os.path.splitext(file)[0]
                trimmed_filename = f"{filename_wo_ext}_trimmed.avi"
                output_path = os.path.join(root, trimmed_filename)

                extract_center_clip(full_path, output_path)

if __name__ == "__main__":
    train_dir = '/Users/mzeeshan/Documents/PythonProjects/FYP/train'
    process_all_videos_in_dataset(train_dir)
