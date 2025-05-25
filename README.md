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

### 🛰️ 1. Dataset
We used the **Satellite Imagery of Ships** dataset, which contains high-resolution satellite images labeled as either:

- `ship`
- `no-ship`

These images simulate real-world detection tasks in maritime surveillance and defense.

---

### 🔍 2. Super-Resolution using SwinIR

We applied the **SwinIR-Large x4 GAN** model to upscale low-resolution satellite images before classification. This method improves image clarity and may enhance the model's ability to detect small or blurry ships.

- Model used: [`003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth`](https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth)
- Source: [SwinIR GitHub Repository](https://github.com/JingyunLiang/SwinIR)

---

## 🖼️ 3. Classification Models

### ✅ ResNet50

- Implemented with transfer learning (from `torchvision.models`).
- Trained separately on raw and enhanced images.
- Results:

| Model Variant               | Input Type         | Accuracy | Precision (ship) | Recall (ship) | F1-score (ship) |
|----------------------------|--------------------|----------|------------------|---------------|-----------------|
| ResNet50                   | Raw images         | **99.50%** | 1.00             | 0.98          | 0.99            |
| ResNet50                   | Super-resolved     | **98.97%** | 0.98             | 0.98          | 0.98            |

<details>
<summary><strong>ResNet50 – Confusion Matrices</strong></summary>

**Raw Images:**

- No-ship: 292 correct, 0 false positives  
- Ship: 106 correct, 2 false negatives

**Super-Resolution Images:**

- No-ship: 281 correct, 2 false positives  
- Ship: 102 correct, 2 false negatives

</details>

📌 **Observation:** Although super-resolution enhances visual clarity, in our case, classification performed slightly better on raw images with ResNet50.

---

### ✅ YOLOv8n Classification

- Model: `yolov8n-cls` from [Ultralytics](https://github.com/ultralytics/ultralytics)
- Approach:
  - Trained on raw images
  - Trained on enhanced images
  - Compared performance with frozen backbone layers
- Final performance metrics are available in the YOLOv8 notebooks.

📌 **Note:** YOLOv8n offers a lightweight and fast solution suitable for edge deployment scenarios.

---

## 🧪 Tools & Environment

- Python 3.10+
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- SwinIR (Image Super-Resolution)
- Google Colab / Jupyter Notebook

---

## 📌 Conclusion

This project shows that **super-resolution can enhance satellite image classification**, but it's not always guaranteed to improve model performance. In our experiments:

- **ResNet50** achieved best accuracy on **raw images** (99.50%)
- Super-resolution slightly decreased accuracy but maintained high precision and recall
- **YOLOv8n** served as a fast and scalable classifier for real-time or embedded systems

These insights are valuable for future satellite imagery analysis and real-world maritime detection pipelines.

---

## 👩‍💻 Authors

- Tzuf Lahan   
- Ayelet Hashahar Cohen   
- Adi Daniel   

