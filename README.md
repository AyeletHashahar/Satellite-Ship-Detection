# 🚢 Satellite Ship Classification using Super-Resolution and Deep Learning 🛰️

This project focuses on **ship classification from satellite images** using deep learning techniques, and evaluates how **image super-resolution** influences classification performance. We leverage two powerful model families — **ResNet50** and **YOLOv8** — to classify images both in their raw form and after enhancement.

> 📦 Dataset: [Satellite Imagery of Ships (Kaggle)](https://www.kaggle.com/datasets/apollo2506/satellite-imagery-of-ships)

---

## 📁 Project Structure

- `EDA/EDA.ipynb` – Initial data exploration and class distribution
- `Enhanced resolution/Enhanced_Resolution.ipynb` – Image enhancement using SwinIR super-resolution
- `ResNet50/RegularPic.ipynb` – Baseline classification using ResNet50
- `ResNet50/SuperResu.ipynb` – ResNet50 classification on enhanced (super-resolved) images
- `YOLO8/yolov8n_cls.ipynb` – YOLOv8n classification on raw images
- `YOLO8/yolov8n_cls_update.ipynb` – YOLOv8n classification on enhanced images (with and without frozen layers)

---

## 🧠 Project Overview

## 🛰️ Dataset

The dataset contains high-resolution satellite images labeled into two classes:

- `ship`
- `no-ship`

These images simulate real-world maritime detection tasks.

---

## 🧪 Super-Resolution (SwinIR)

To improve image clarity, we applied the **SwinIR-Large x4 GAN** model to upscale low-resolution satellite images before classification.

- Model: [`003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth`](https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth)
- Source: [SwinIR GitHub](https://github.com/JingyunLiang/SwinIR)

---

## 🧠 Classification Models

### ✅ ResNet50

| Model Variant               | Input Type         | Accuracy | Precision (ship) | Recall (ship) | F1-score (ship) |
|----------------------------|--------------------|----------|------------------|---------------|-----------------|
| ResNet50                   | Raw images         | **99.50%** | 1.00             | 0.98          | 0.99            |
| ResNet50                   | Super-resolved     | **98.97%** | 0.98             | 0.98          | 0.98            |

**Raw Images Confusion Matrix:**
- No-ship: 292 correct, 0 false positives  
- Ship: 106 correct, 2 false negatives

**Super-Resolution Confusion Matrix:**
- No-ship: 281 correct, 2 false positives  
- Ship: 102 correct, 2 false negatives

📌 **Insight:** While super-resolution enhances clarity, the raw images gave slightly better results with ResNet50.

---

### ✅ YOLOv8n Classification

We used Ultralytics' YOLOv8n model (`yolov8n-cls`) for lightweight image classification.

#### YOLOv8n on Raw Images

| Metric      | Ship   | No-ship |
|-------------|--------|---------|
| Precision   | 1.00   | 0.93    |
| Recall      | 0.97   | 1.00    |
| F1-score    | 0.99   | 0.96    |

- **Test Accuracy:** **98.00%**
- **Confusion Matrix:**
  - Ship: 100 correct, 0 false positives
  - No-ship: 292 correct, 8 false negatives

#### YOLOv8n with Frozen Weights

We evaluated the effect of **freezing the early layers** of the model.

| Mode              | Accuracy |
|-------------------|----------|
| Normal Training   | 97.25%   |
| Frozen Weights    | 97.25%   |

📌 **Insight:** Freezing early layers did **not reduce** performance, suggesting pretrained features are robust enough for this binary classification task.

---

## ⚙️ Tools Used

- Python 3.10+
- PyTorch
- Ultralytics YOLOv8
- SwinIR (for super-resolution)
- OpenCV, NumPy, scikit-learn
- Google Colab for training and evaluation

---

## 📌 Conclusion

This project demonstrates:

- ResNet50 achieved best performance on raw images (99.50% accuracy)
- Super-resolution improved visual clarity but did not significantly boost performance
- YOLOv8n performed robustly on raw images (98% accuracy)
- Freezing weights in YOLOv8n had no negative impact on accuracy

The results suggest that **lightweight models like YOLOv8n can match heavy models like ResNet50** in binary classification tasks involving satellite imagery.

---
## 👩‍💻 Authors

- Tzuf Lahan   
- Ayelet Hashahar Cohen   
- Adi Daniel   

