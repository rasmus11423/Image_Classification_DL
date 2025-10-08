import torch
from typing import List, Tuple, Optional
from PIL import Image

# Dataset utilities (standard or stratified split)
class FoodDataset(torch.utils.data.Dataset):
    def __init__(self, items: List[Tuple[str, int]], transform=None, class_names: Optional[List[str]] = None):
        self.items = items
        self.transform = transform
        self.class_names = class_names

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label