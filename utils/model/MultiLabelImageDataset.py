import os
import torch
from torch.utils.data import Dataset
from PIL import Image



class MultiLabelImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        image = Image.open(img_name)  # Convert to RGB
        label_gender = self.df.iloc[idx, 1]
        label_race = self.df.iloc[idx, 2]
        label_age = self.df.iloc[idx, 3]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label_gender), torch.tensor(label_race), torch.tensor(label_age)
