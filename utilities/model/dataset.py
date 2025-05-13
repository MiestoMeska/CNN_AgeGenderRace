import os
import torch
from torch.utils.data import Dataset

class PreprocessedDataset(Dataset):
    """
    Dataset that loads preprocessed .pt tensors for images,
    along with gender and age labels.
    """
    def __init__(self, df, preprocessed_dir):
        self.df = df.reset_index(drop=True)
        self.pre_dir = preprocessed_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_tensor = torch.load(os.path.join(self.pre_dir, row['filename'] + '.pt'))

        gender = torch.tensor(row['gender'], dtype=torch.long)
        age = torch.tensor(row['age'], dtype=torch.float32)

        return img_tensor, gender, age
