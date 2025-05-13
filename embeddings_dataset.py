import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    def __init__(self, folder, number_of_arrays=1):
        """
        Dataset that loads .npy embedding arrays from 'folder' on demand.
        Each .npy file must match pattern b####.npy.

        Args:
            folder (str): Path to the directory containing .npy files.
            number_of_arrays (int): Number of files to load and concatenate per sample.
        """
        self.folder = folder
        self.k = number_of_arrays
        # Gather all .npy files matching the b0000.npy pattern
        pattern = os.path.join(folder, "b[0-9][0-9][0-9][0-9].npy")
        self.file_paths = sorted(glob.glob(pattern))
        assert len(self.file_paths) >= self.k, f"Not enough .npy files in {folder}"
        # Group file paths into non-overlapping chunks of size k
        num_groups = len(self.file_paths) // self.k
        self.groups = [
            self.file_paths[i*self.k : (i+1)*self.k]
            for i in range(num_groups)
        ]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        file_group = self.groups[idx]
        arrays = []
        for path in file_group:
            arr = np.load(path)  # shape (2432, 101, 128)
            arr = arr.reshape(-1, arr.shape[-1])  # (2432*101, 128)
            arrays.append(arr)
        batch = np.concatenate(arrays, axis=0)  # (k*2432*101, 128)
        return torch.from_numpy(batch).float() 