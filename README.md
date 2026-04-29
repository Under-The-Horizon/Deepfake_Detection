# Deepfake_Detection

This README is structured to highlight the sophisticated hybrid architecture of your project. It balances the high-level vision with the specific technical implementation details that recruiters and collaborators look for in a deepfake detection repository.

## 🚀 Key Features
* **Hybrid Architecture:** Combines CNN stems for efficient downsampling with Swin Transformer blocks for global feature extraction.
* **Contrastive Learning:** Utilizes **Supervised Contrastive Loss** to maximize the distance between "real" and "fake" feature embeddings.
* **Modular Pipeline:** Includes dedicated modules for generative latent extraction and Siamese contrastive classification.
* **Real-time Inference:** Integrated with a **Gradio web interface** for easy video testing and visualization.
* **Lightweight Efficiency:** Optimized sequence processing to balance high accuracy with lower computational overhead.

## 🏗️ Architecture Overview
The model follows a specialized pipeline designed to identify the subtle "generative" signatures of deepfakes:
1.  **CViT Stem:** Efficient early-stage downsampling and feature extraction.
2.  **Swin Transformer Blocks:** Captures hierarchical spatial relationships within frames.
3.  **1D-CNN Classification Head:** Processes temporal sequences to detect inconsistencies across the video duration.
4.  **DeepFeatureX-SN:** A Siamese Network component that enhances discriminative power through feature comparison.

## 📊 Performance
The model was trained and evaluated on the **Celeb-DF-v2** dataset, a standard benchmark for high-quality deepfake detection.
* **Accuracy:** ~97%
* **Loss Function:** Supervised Contrastive Loss
* **Optimization:** Balanced for both detection precision and inference speed.

## 🛠️ Tech Stack
* **Frameworks:** PyTorch, Torchvision
* **Interface:** Gradio
* **Data Processing:** OpenCV, NumPy
* **Architecture:** CNN Stems, Swin Transformers, 1D-CNN

## 📂 Project Structure
```text
├── models/             # GenConViT & CViT architecture definitions
├── utils/              # Video preprocessing and frame extraction
├── weights/            # Pre-trained model checkpoints
├── app.py              # Gradio web interface
├── train.py            # Training script with Contrastive Loss
└── requirements.txt    # Dependencies
```

