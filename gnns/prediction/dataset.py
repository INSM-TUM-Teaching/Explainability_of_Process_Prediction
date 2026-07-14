"""On-disk graph dataset used when graphs are pre-materialized as ``.pt`` files."""
import os

import torch
from torch.utils.data import Dataset


class GraphFolderDataset(Dataset):

    def __init__(self, folder: str):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Dataset folder not found: {folder}")

        self.folder = folder
        self.files = sorted(
            [f for f in os.listdir(folder) if f.endswith(".pt")],
            key=lambda x: int(os.path.splitext(x)[0]),
        )
        if not self.files:
            raise FileNotFoundError(f"No .pt graphs found in {folder}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.folder, self.files[idx])
        graph = torch.load(path, map_location="cpu", weights_only=False)
        return graph
