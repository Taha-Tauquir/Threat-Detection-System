import os

def delete_trimmed_videos(root_dir):
    deleted_files = []

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            # if file.endswith(".avi") and "_trimmed" in file:
            if file.endswith(".csv"):
            # and "_trimmed" in file:
                file_path = os.path.join(subdir, file)
                try:
                    os.remove(file_path)
                    deleted_files.append(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

    print(f"\nTotal deleted files: {len(deleted_files)}")


root_directory = "/Users/mzeeshan/Documents/PythonProjects/NewRppg/train"
delete_trimmed_videos(root_directory)
