# 🛰️ Satellite Ship Detection with Super Resolution and Deep Learning 🚢

This project explores the task of **ship classification in satellite images** by combining:

- 🧪 Exploratory Data Analysis (EDA)
- 🔁 Image enhancement via **Super Resolution (SwinIR)**
- 🧠 Classification using **ResNet50** and **YOLOv8**
- 📊 Performance comparison before and after enhancement

---

## 📁 Project Structure

📦 Satellite-Ship-Detection/
├── EDA/
│ └── EDA.ipynb # Visual analysis of class distribution, brightness, color, edges
├── Enhanced resolution/
│ └── Enhanced_Resolution.ipynb # Super Resolution using SwinIR (x4 GAN)
├── ResNet50/
│ ├── ResNet50.ipynb # Train ResNet50 on original images
│ └── SuperResu.ipynb # Train ResNet50 on enhanced images
├── YOLO8/
│ ├── yolov8n_cls.ipynb # YOLOv8 for classification (original images)
│ └── yolov8n_cls_update.ipynb # YOLOv8 for classification (enhanced images)
└── README.md # Project overview and documentation


---

## 🎯 Project Goals

- Classify satellite images into `ship` / `no-ship`
- Improve image clarity using **SwinIR Super Resolution**
- Compare performance of models trained on original vs enhanced data

---

## 🔍 Dataset

We used the [Satellite Imagery of Ships dataset](https://www.kaggle.com/datasets/apollo2506/satellite-imagery-of-ships) from Kaggle.  
Each image is 80×80 pixels and labeled as `ship` or `no-ship`.

---

## 🚀 Models and Experiments

### 📊 EDA

Conducted to understand:
- Class imbalance
- Brightness differences
- Color channel distributions
- Edge detection patterns

### 🧠 ResNet50 (Original)

- Accuracy: **99.5%**
- F1 (ship): **0.99**
- No overfitting (loss curves stable)

### ✨ SwinIR Super Resolution

We used the **SwinIR-Large x4 GAN** model:
- GitHub: [SwinIR](https://github.com/JingyunLiang/SwinIR)
- Paper: [arXiv:2108.10257](https://arxiv.org/abs/2108.10257)

### 🧠 ResNet50 (Enhanced)

- Accuracy: **98.97%**
- F1 (ship): **0.98**
- Generalization remains strong

### ⚡ YOLOv8 Classification

- Trained using `ultralytics` on both original and enhanced images
- Similar high performance across both

---

## ✅ Results Summary

| Model     | Data Type | Accuracy | F1 (Ship) | Overfitting |
|-----------|-----------|----------|-----------|-------------|
| ResNet50  | Original  | 99.5%    | 0.99      | ❌ Low       |
| ResNet50  | Enhanced  | 98.97%   | 0.98      | ❌ Low       |
| YOLOv8n   | Both      | ~99%     | ~0.98–0.99| ❌ Low       |

---

## 📦 Installation (Google Colab)

Install required packages:

```bash
pip install timm opencv-python ultralytics

🙌 Acknowledgments
SwinIR – Swin Transformer for Image Restoration

Ultralytics – YOLOv8

Kaggle Dataset


