# Satellite Ship Image Super-Resolution using SwinIR 🚢🛰️✨

This project applies **SwinIR (Swin Transformer for Image Restoration)** to enhance the resolution of satellite images, specifically focusing on **ship detection and visualization**. It leverages pre-trained SwinIR models to upscale images and improve visual clarity and detail—essential for analyzing low-resolution satellite captures.

The model used for enhancement is the **SwinIR-Large x4 GAN** model, available [here](https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth).

## 🔍 What is SwinIR?
SwinIR is a state-of-the-art image restoration model based on the Swin Transformer. It excels at tasks like:
- Super-resolution
- Denoising
- Compression artifact reduction

For more technical details, check out the [SwinIR GitHub repository](https://github.com/JingyunLiang/SwinIR)

The original paper:**"SwinIR: Image Restoration Using Swin Transformer"** – [arXiv:2108.10257](https://arxiv.org/abs/2108.10257).

---

## 🚀 Project Goals
- Improve the clarity of satellite images for ship detection and maritime surveillance.
- Use the enhanced images generated by SwinIR to **train a CNN model**, resulting in **improved classification accuracy** for ship detection tasks.

---

## 🖼️ Before vs. After Comparisons

Below are some visual comparisons of original vs. enhanced images using **SwinIR-Large**:

![Image](https://github.com/user-attachments/assets/b7d8371f-069c-4ca5-8e2f-2df6efac8846)


![Image](https://github.com/user-attachments/assets/888f1539-9d48-4ffd-b725-dcd051771092)


![Image](https://github.com/user-attachments/assets/6978c0f6-4d1a-4d19-b98d-87af19bfe7fc)


## 🛠️ Requirements
This implementation runs in **Google Colab** and installs:
- `timm`
- `OpenCV`
- Pre-trained SwinIR model weights

  ## 🙌 Acknowledgments
- [Jingyun Liang et al.](https://github.com/JingyunLiang/SwinIR) for the original SwinIR implementation.
- The open-source community for the tools used in this notebook.
