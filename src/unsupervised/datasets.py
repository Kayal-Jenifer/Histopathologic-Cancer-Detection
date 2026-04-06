from pathlib import Path

import numpy as np
from PIL import Image

from torch.utils.data import Dataset


class ImageIdDataset(Dataset):
    def __init__(self, train_dir: Path, ids, transform):
        self.train_dir = train_dir
        self.ids = list(ids)
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img = Image.open(self.train_dir / f"{img_id}.tif").convert("RGB")
        x = self.transform(img)
        return x, img_id


class LabeledImageDataset(Dataset):
    def __init__(self, train_dir: Path, ids, labels, transform):
        self.train_dir = train_dir
        self.ids = list(ids)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img = Image.open(self.train_dir / f"{img_id}.tif").convert("RGB")
        x = self.transform(img)
        y = int(self.labels[idx])
        return x, y
