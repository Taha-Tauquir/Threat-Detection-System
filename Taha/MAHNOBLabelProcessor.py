import os
import pandas as pd
import numpy as np
import shutil
import random

class MAHNOBLabelProcessor:
    def __init__(self, dataset_dir, label_save_dir, train_ratio=0.8):
        self.dataset_dir = dataset_dir
        self.label_save_dir = label_save_dir
        self.train_ratio = train_ratio

        os.makedirs(self.label_save_dir, exist_ok=True)

    def generate_label_npy(self, csv_path, participant_id):
        df = pd.read_csv(csv_path, header=None)
        assert df.shape[0] == 900, f"Expected 900 rows, found {df.shape[0]} in {csv_path}"

        label_data = df.iloc[:, 1].values
        filename = f"{str(participant_id).zfill(4)}_label.npy"
        save_path = os.path.join(self.label_save_dir, filename)

        np.save(save_path, label_data)
        print(f" Saved label: {save_path}")

    def iterate_csv_and_generate_labels(self):
        for participant_id in sorted(os.listdir(self.dataset_dir)):
            session_path = os.path.join(self.dataset_dir, participant_id)
            if not os.path.isdir(session_path):
                continue

            for file in os.listdir(session_path):
                if file.endswith("_30hz.csv"):
                    csv_path = os.path.join(session_path, file)
                    try:
                        self.generate_label_npy(csv_path, participant_id)
                    except Exception as e:
                        print(f"⚠️ Error in {participant_id}: {e}")
                    break  # Assumes one label CSV per participant


    def split_labels_and_inputs(self,input_dir, train_ratio=0.8):
        # Extract all base IDs (e.g., 0426 from 0426_label.npy)
        base_ids = sorted([
            f.split('_')[0] for f in os.listdir(input_dir)
            if
            f.endswith('_label.npy') and os.path.exists(os.path.join(input_dir, f.replace('_label.npy', '_input.npy')))
        ])

        random.shuffle(base_ids)
        train_count = int(len(base_ids) * train_ratio)

        train_ids = base_ids[:train_count]
        test_ids = base_ids[train_count:]

        # Create output folders
        for folder in ['train', 'test']:
            os.makedirs(os.path.join(input_dir, folder), exist_ok=True)
            os.makedirs(os.path.join(input_dir, folder), exist_ok=True)

        # Move files based on split
        for file_id in train_ids:
            shutil.move(os.path.join(input_dir, f"{file_id}_label.npy"),
                        os.path.join(input_dir, "train", f"{file_id}_label.npy"))
            shutil.move(os.path.join(input_dir, f"{file_id}_input.npy"),
                        os.path.join(input_dir, "train", f"{file_id}_input.npy"))

        for file_id in test_ids:
            shutil.move(os.path.join(input_dir, f"{file_id}_label.npy"),
                        os.path.join(input_dir, "test", f"{file_id}_label.npy"))
            shutil.move(os.path.join(input_dir, f"{file_id}_input.npy"),
                        os.path.join(input_dir, "test", f"{file_id}_input.npy"))

        print(f"\nTotal Pairs: {len(base_ids)} | Train: {len(train_ids)} | Test: {len(test_ids)}")
        print(f"Label Train Folder: {os.path.join(input_dir, 'train')}")
        print(f"Input Train Folder: {os.path.join(input_dir, 'train')}")

def create_train_test_csv_lists_for_physnet(cached_dir, output_dir=None):
    """
    Creates train_list.csv and test_list.csv for PhysNet.
    Each CSV contains only the base names (without .npy extension)
    of input NPY files found in 'train/' and 'test/' subfolders.

    Parameters:
        cached_dir (str): The parent folder that contains 'train' and 'test' subfolders.
        output_dir (str): Directory to save the CSVs. Defaults to `cached_dir/filelists/`.
    """
    train_folder = os.path.join(cached_dir, "train")
    test_folder = os.path.join(cached_dir, "test")

    if output_dir is None:
        output_dir = os.path.join(cached_dir, "filelists")
    os.makedirs(output_dir, exist_ok=True)

    def get_input_ids(folder):
        return sorted([
            os.path.splitext(f)[0]  # removes .npy
            for f in os.listdir(folder)
            if f.endswith("_input.npy")
        ])

    train_ids = get_input_ids(train_folder)
    test_ids = get_input_ids(test_folder)

    pd.DataFrame(train_ids).to_csv(os.path.join(output_dir, "train_list.csv"), index=False, header=False)
    pd.DataFrame(test_ids).to_csv(os.path.join(output_dir, "test_list.csv"), index=False, header=False)

    print(f"✅ PhysNet-compatible train_list.csv and test_list.csv created in: {output_dir}")
    print(f"Train samples: {len(train_ids)} | Test samples: {len(test_ids)}")
