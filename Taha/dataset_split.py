import os
import shutil
import random

dataset_dir = '/Users/mzeeshan/Documents/PythonProjects/NewRppgHundred/Sessions'  
train_dir = '/Users/mzeeshan/Documents/PythonProjects/NewRppgHundred/train' 
test_dir = '/Users/mzeeshan/Documents/PythonProjects/NewRppgHundred/test'


def split_dataset_randomly(dataset_dir, train_dir, test_dir, split_ratio=0.8):
    all_folders = sorted([f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))])
    random.shuffle(all_folders)
    train_size = int(len(all_folders) * split_ratio)
    train_folders = all_folders[:train_size]
    test_folders = all_folders[train_size:]
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    def copy_folders(folders, target_dir):
        for folder in folders:
            folder_path = os.path.join(dataset_dir, folder)
            shutil.copytree(folder_path, os.path.join(target_dir, folder))

    copy_folders(train_folders, train_dir)
    copy_folders(test_folders, test_dir)

    print(f"Dataset split: {len(train_folders)} folders for training, {len(test_folders)} folders for testing.")
    print(f"Training data is stored in {train_dir}")
    print(f"Testing data is stored in {test_dir}")

if __name__ == "__main__":
    split_dataset_randomly(dataset_dir, train_dir, test_dir)