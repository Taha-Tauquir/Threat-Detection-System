import os
import shutil

# Update this to your actual path
TRAIN_DIR = '/Users/mzeeshan/Documents/PythonProjects/NewRppg/train'

def delete_target_folders(base_dir):
    for session_id in sorted(os.listdir(base_dir)):
        session_path = os.path.join(base_dir, session_id)
        if not os.path.isdir(session_path):
            continue

        print(f"\nChecking: {session_path}")
        for subfolder in os.listdir(session_path):
            subfolder_path = os.path.join(session_path, subfolder)

            if subfolder == "inconsistent_frames":
                print(f"  Deleting folder: {subfolder_path}")
                shutil.rmtree(subfolder_path)

            elif subfolder.endswith("trimmed_frames"):
                print(f"  Deleting folder: {subfolder_path}")
                shutil.rmtree(subfolder_path)

if __name__ == "__main__":
    delete_target_folders(TRAIN_DIR)
