"""

Author: Virendra Badgotya
Team Name: VIR
Team Members: Virendra Badgotya(Solo)
Leaderboard Rank: 52

"""

import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Label encoding
def load_and_split_data(labels_csv, train_dir, test_size=0.15, random_state=42):
    df = pd.read_csv(labels_csv)
    df['image'] = df['image_id']
    label_mapping = {label: idx for idx, label in enumerate(df['soil_type'].unique())}
    df['label'] = df['soil_type'].map(label_mapping)

    train_df, val_df = train_test_split(
        df, test_size=test_size, stratify=df['label'], random_state=random_state
    )
    return train_df, val_df, label_mapping

# Image transforms
def get_transforms():
    return {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
    }

# Custom dataset
class SoilDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, is_test=False):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx]['image']
        img_path = os.path.join(self.img_dir, image_id)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, image_id
        else:
            label = self.df.iloc[idx]['label']
            return image, label

# Create loaders
def create_loaders(train_df, val_df, train_dir, transforms_dict, batch_size=32):
    train_dataset = SoilDataset(train_df, train_dir, transform=transforms_dict['train'])
    val_dataset = SoilDataset(val_df, train_dir, transform=transforms_dict['val'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def preprocessing():
    print("This is the file for preprocessing")
  return 0
