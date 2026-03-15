import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class HandwrittenDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Create label list and map string labels to integers
        self.label_list = sorted(set(self.annotations['label']))
        self.label_map = {label: idx for idx, label in enumerate(self.label_list)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = self.annotations.iloc[idx, 0]  # e.g. 'img/img001-001.png'
        label_str = self.annotations.iloc[idx, 1]

        # Fix path to image file
        relative_path = os.path.normpath(img_name)
        filename_only = os.path.basename(relative_path)
        img_path = os.path.join(self.img_dir, filename_only)

        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        label = self.label_map[label_str]  # Convert label string to numeric index

        return image, label
