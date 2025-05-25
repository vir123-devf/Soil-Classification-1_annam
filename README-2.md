#  Soil Classification Hackathon Challenge 2 - IIT Ropar | Annam.ai


This repository contains my solution to the **Soil Image Classification Challenge 2** organized by **Annam.ai at IIT Ropar**. The goal was to classify whether an image is of soil or not using machine learning and computer vision techniques.

> ✅ **Approach Used**: Feature extraction using **ResNet-18** + **One-Class SVM**

---
## 👥 Team Details

**Team Name:** VIR

**Team Members:**(Solo)
- Virendra badgotya *(Team Lead)*


**Leaderboard Rank:** 35 (Part 2)  
**Kaggle Score:** 0.9222🎯

---

## 📂 Dataset Structure

The dataset consists of images split into two folders and accompanying CSV files:

```
soil_competition-2025/
├── train/
├── test/
├── train_labels.csv
├── test_ids.csv
├── sample_submission.csv
```

* `train/`: 1222 labeled soil images
* `test/`: 967 unlabeled images for prediction
* `train_labels.csv`: Maps each `image_id` to a `soil_type`
* `test_ids.csv`: Lists the image IDs of test images

---

## 🧠 Model Pipeline

### 🔍 Feature Extraction

* A pretrained **ResNet-18** model is used to extract 512-dimensional feature vectors from each image.
* The final classification head (`fc`) of ResNet18 is replaced with an identity layer.

### 🧼 Preprocessing

* Resize images to **224×224**
* Normalize using mean and std of `[0.5, 0.5, 0.5]`
* Apply `transforms.ToTensor()`

### 🔎 One-Class SVM

* Fit a **One-Class SVM** on training features (which only includes soil images)
* Use RBF kernel with `nu=0.1` to model the decision boundary around the “soil” class
* Predict whether test images belong to the soil class (`1`) or not (`0`)

---

## 🚀 How to Run

1. Clone the repository and install dependencies
2. Place the dataset in the expected structure
3. Run the notebook or script for feature extraction and SVM prediction
4. Generates a `submission.csv` file ready for submission

---

## 📈 Evaluation

* **Metric**: **F1-Score**
* The challenge uses **binary classification**:

  * `1` → Soil
  * `0` → Non-Soil
* Final score balances precision and recall
* F1 score = **0.9222**
---

## 📦 Submission Format

A `submission.csv` file with the following structure: 

```csv
image_id,label
abc123.jpg,1
xyz789.jpg,0
```

---

## 📑 Requirements

* Python ≥ 3.8
* PyTorch
* torchvision
* scikit-learn
* pandas
* tqdm
* PIL

Install via:

```bash
pip install torch torchvision scikit-learn pandas tqdm pillow
```

---

## 🤝 Acknowledgements

* Hosted by: [IIT Ropar](https://www.iitrpr.ac.in)
* Organised by: [Annam.ai](https://annam.ai)
* Dataset: Provided by the Soil Image Classification Challenge team
  
---
## 🧑‍💻 Author

**Virendra Badgotya**
* AI/ML Enthusiast | B.Tech @ SVNIT
  
  🔗 [LinkedIn](https://www.linkedin.com/in/virendra-badgotya/) | [GitHub](https://github.com/vir123-devf)



