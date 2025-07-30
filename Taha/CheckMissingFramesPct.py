import os

EXPECTED_FRAME_COUNT = 900
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']

def count_images(folder_path):
    """Count image files in a given folder."""
    return sum(1 for f in os.listdir(folder_path)
               if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS)

def analyze_sessions(root_dir):
    print(f"{'Session':<10} {'Missing Frames':>15} {'Missing (%)':>15}")
    print("-" * 45)

    for session_name in os.listdir(root_dir):
        session_path = os.path.join(root_dir, session_name)
        if os.path.isdir(session_path):
            inconsistent_path = os.path.join(session_path, "inconsistent_frames_folder")
            
            if os.path.exists(inconsistent_path):
                present = count_images(inconsistent_path)
                missing =  present
                missing_pct = (missing / EXPECTED_FRAME_COUNT) * 100
                print(f"{session_name:<10} {missing:>15} {missing_pct:>14.2f}%")



if __name__ == "__main__":
    BASE_PATH = "/Users/mzeeshan/Documents/PythonProjects/NewRppgHundred/train"
    analyze_sessions(BASE_PATH)