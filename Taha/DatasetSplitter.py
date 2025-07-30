# dataset_splitter.py

import os
import shutil
import random

def split_npy_files(source_dir, train_ratio=0.8):
    """
    Splits .npy files in the given directory into 'train' and 'test' folders
    based on the given ratio.

    Parameters:
        source_dir (str): Path to directory containing .npy files.
        train_ratio (float): Proportion of files to put in 'train'. Default is 0.8.
    """
    train_dir = os.path.join(source_dir, "train")
    test_dir = os.path.join(source_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    npy_files = [f for f in os.listdir(source_dir) if f.endswith(".npy")]

    if not npy_files:
        print(f"No .npy files found in {source_dir}")
        return

    random.shuffle(npy_files)
    train_count = int(len(npy_files) * train_ratio)
    train_files = npy_files[:train_count]
    test_files = npy_files[train_count:]

    for f in train_files:
        shutil.move(os.path.join(source_dir, f), os.path.join(train_dir, f))
    for f in test_files:
        shutil.move(os.path.join(source_dir, f), os.path.join(test_dir, f))

    print(f"Total files: {len(npy_files)}")
    print(f"Train files: {len(train_files)} -> {train_dir}")
    print(f"Test files:  {len(test_files)} -> {test_dir}")
