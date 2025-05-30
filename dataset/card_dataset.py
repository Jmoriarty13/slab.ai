import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from albumentations import Compose, Resize, Normalize, HorizontalFlip
from albumentations.pytorch import ToTensorV2

class DualScaleCardDataset(Dataset):
    def __init__(self, df_path, image_dir, crops_dir, global_size=(384, 384), local_size=(128, 128), transform=True):
        self.data = pd.read_csv(df_path)
        self.image_dir = image_dir
        self.crops_dir = crops_dir
        self.global_size = global_size
        self.local_size = local_size
        self.transform = transform

        self.global_transform = Compose([
            Resize(*global_size),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

        self.local_transform = Compose([
            Resize(*local_size),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_dir, row["image"])
        label = torch.tensor(row[["centering", "corners", "edges", "surface", "overall"]].values, dtype=torch.float32)

        # Load and transform global image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        global_img = self.global_transform(image=img)["image"]

        # Load and transform local patches
        local_patches = []
        for i in range(1, 27):  # Assuming 26 local patches
            patch_name = f"{row['id']}_patch{i}.png"
            patch_path = os.path.join(self.crops_dir, patch_name)
            if os.path.exists(patch_path):
                patch = cv2.imread(patch_path)
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                patch = self.local_transform(image=patch)["image"]
                local_patches.append(patch)
            else:
                local_patches.append(torch.zeros(3, *self.local_size))  # Placeholder

        local_stack = torch.stack(local_patches)  # Shape: [26, 3, 128, 128]

        return {
            "global_img": global_img,
            "local_patches": local_stack,
            "label": label
        }
