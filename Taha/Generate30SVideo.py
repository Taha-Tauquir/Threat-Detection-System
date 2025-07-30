import os
import ffmpeg

import os
import ffmpeg

import os
import ffmpeg

def trim_middle_30s_ffmpeg(input_path, output_dir=None):
    base_name = os.path.basename(input_path)
    name, _ = os.path.splitext(base_name)

    try:
        probe = ffmpeg.probe(input_path)
        duration = float(probe['format']['duration'])
    except Exception as e:
        print(f"❌ Failed to probe {input_path}: {e}")
        return

    print(f"\n📹 Video: {base_name}")
    print(f"Duration: {duration:.2f} seconds")

    if duration < 30:
        print(f"⚠️ Skipping {base_name}: too short")
        return

    start_time = max((duration / 2) - 15, 0)
    print(f"⏩ Trimming from {start_time:.2f}s to {start_time + 30:.2f}s")

    # Output as high-quality .mp4
    trimmed_name = f"{name}_cropped.mp4"
    output_path = os.path.join(output_dir or os.path.dirname(input_path), trimmed_name)

    try:
        (
            ffmpeg
            .input(input_path, ss=start_time, t=30)
            .output(output_path,
                    vcodec='libx264',
                    crf=18,                # change to 0 for lossless
                    preset='fast',
                    acodec='aac')
            .run(overwrite_output=True, quiet=True)
        )
        print(f"✅ Trimmed video saved: {output_path}")
    except ffmpeg.Error as e:
        print(f"❌ FFmpeg error: {e.stderr.decode()}")




def process_all_videos(root_dir):
    for folder_name in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.lower().endswith(".avi") and "_cropped" not in file.lower():
                    video_path = os.path.join(subdir_path, file)
                    cropped_path = video_path.replace(".avi", "_cropped.avi")

                    if os.path.exists(cropped_path):
                        print(f"✅ Skipping (already cropped): {cropped_path}")
                        continue

                    print(f"🔄 Processing: {video_path}")
                    try:
                        trim_middle_30s_ffmpeg(video_path)
                    except Exception as e:
                        print(f"❌ Error processing {video_path}: {e}")

# Example usage:
# if __name__ == "__main__":
#     input_videos_path = "/path/to/your/folder"
#     process_all_videos_ffmpeg(input_videos_path)
