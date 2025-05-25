"""

Author: Annam.ai IIT Ropar
Team Name: VIR
Team Members: Virendra Badgotya(Solo)
Leaderboard Rank: 35


"""

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from preprocessing import get_dataloaders, extract_features

def main():
    train_loader, test_loader = get_dataloaders()

    # Extract features
    train_features, _ = extract_features(train_loader)
    test_features, test_ids = extract_features(test_loader)

    # Normalize features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    # Train One-Class SVM on soil-only training data
    svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    svm.fit(train_features)

    # Predict on test set (1 = soil, -1 = non-soil)
    svm_preds = svm.predict(test_features)
    binary_preds = [1 if p == 1 else 0 for p in svm_preds]

    # Save submission file
    submission = pd.DataFrame({
        'image_id': test_ids,
        'label': binary_preds
    })
    submission.to_csv('submission.csv', index=False)
    print("Submission file saved as 'submission.csv'")

if __name__ == "__main__":
    main()

def postprocessing():
    print("This is the file for postprocessing")
  return 0
