"""

Author: Annam.ai IIT Ropar
Team Name: VIR
Team Members: Virendra Badgotya(Solo)
Leaderboard Rank: 35

"""

import os
import numpy as np
import pandas as pd
import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# Paths - Adjust these if running locally or elsewhere
train_csv = '/kaggle/input/soil-classification-part-2/soil_competition-2025/train_labels.csv'
test_csv = '/kaggle/input/soil-classification-part-2/soil_competition-2025/test_ids.csv'
train_dir = '/kaggle/input/soil-classification-part-2/soil_competition-2025/train'
test_dir = '/kaggle/input/soil-classification-part-2/soil_competition-2025/test'

# Read CSVs
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# Transform: Resize and normalize for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # Normalize to [-1, 1]
])

# Custom Dataset
class SoilDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx]['image_id']
        image_path = os.path.join(self.img_dir, image_id)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, image_id

def get_dataloaders(batch_size=32):
    train_dataset = SoilDataset(train_df, train_dir, transform)
    test_dataset = SoilDataset(test_df, test_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Load pretrained ResNet18 and remove classifier head
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet18(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device)
resnet.eval()

def extract_features(dataloader):
    features = []
    ids = []
    with torch.no_grad():
        for images, image_ids in tqdm(dataloader):
            images = images.to(device)
            feats = resnet(images).cpu().numpy()
            features.append(feats)
            ids.extend(image_ids)
    return np.vstack(features), ids

def preprocessing():
    print("This is the file for preprocessing")
  return 0


