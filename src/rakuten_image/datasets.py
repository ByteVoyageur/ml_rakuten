import warnings
from pathlib import Path
from typing import Optional, Callable, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class RakutenImageDataset(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: str,
        transform: Optional[Callable]=None,
        label_col: str = "prdtypecode"
    ):
        self.image_dir = Path(image_dir)
        self.dataframe = dataframe
        self.transform = transform

        unique_labels = sorted(self.dataframe[label_col].unique())
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.label_mapping = label_to_idx
        self.num_classes = len(unique_labels)

        print("Pre-loading paths into memory...")
        self.image_paths = [
            self.image_dir / f"image_{row['imageid']}_product_{row['productid']}.jpg"
            for _, row in self.dataframe.iterrows()
        ]

        self.labels = dataframe[label_col].map(label_to_idx).tolist()
        print(f"Dataset initialized with {len(self.image_paths)} samples.")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label