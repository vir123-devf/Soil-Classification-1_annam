"""

Author: Virendra Badgotya 
Team Name: VIR
Team Members: Virendra Badgotya(Solo)
Leaderboard Rank: 52

"""
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from preprocessing import SoilDataset, get_transforms

# Predict function
def predict(model, test_df, test_dir, device, label_mapping):
    model.eval()
    transform = get_transforms()['test']
    test_dataset = SoilDataset(test_df, test_dir, transform=transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    predictions, image_ids = [], []

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            predictions.extend(preds)
            image_ids.extend(ids)

    final_labels = [inv_label_mapping[p] for p in predictions]
    return image_ids, final_labels

# Save submission
def save_submission(image_ids, labels, output_file='submission.csv'):
    submission_df = pd.DataFrame({
        'image_id': image_ids,
        'soil_type': labels
    })
    submission_df.to_csv(output_file, index=False)
    print(f"[INFO] Submission saved to {output_file}")

def postprocessing():
    print("This is the file for postprocessing")
  return 0

