# ChestNut 🫘 - Chest X-Ray Disease Detection

> **CheXNet**: Radiologist-Level Disease Detection on Chest X-Rays with Deep Learning

A deep learning application for automated detection and localization of 14 common thoracic diseases in chest X-ray images using DenseNet-121 and PyTorch.

**Live Demo** : [https://chestnut-yrjc.onrender.com](https://chestnut-yrjc.onrender.com/)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Supported Diseases](#supported-diseases)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [API Server](#api-server)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [Author](#author)

---

##  Overview

ChestNut is a deep learning-based diagnostic tool that analyzes chest X-ray images to detect and highlight potential abnormalities. It uses a pre-trained DenseNet-121 model fine-tuned on the NIH ChestX-ray14 dataset to classify 14 common thoracic diseases with high accuracy.

The application provides:
- **Web Interface**: User-friendly interface for uploading and analyzing X-ray images
- **REST API**: FastAPI endpoints for integration with medical systems
- **Visualization**: Class Activation Maps (CAM) to highlight disease regions
- **Multiple Implementations**: Both PyTorch and ONNX optimized versions

---

##  Features

-  **Multi-label Classification**: Detects up to 14 different thoracic diseases simultaneously
-  **Interpretability**: Class Activation Maps (CAM) for visual explanation of predictions
-  **Flexible Inference**: GPU/CPU support with automatic device detection
-  **Web API**: FastAPI-based REST API for easy integration
-  **Production-Ready**: Dockerized deployment with ONNX optimization option
-  **Model Checkpoint Management**: Trained weights and model versioning
-  **Comprehensive Logging**: Debug and track model training/inference
-  **Dataset Tools**: Scripts for dataset preparation and augmentation

---

##  Supported Diseases

The model can detect the following 14 thoracic diseases:

1. **Atelectasis** - Partial or complete lung collapse
2. **Cardiomegaly** - Enlarged heart
3. **Effusion** - Fluid accumulation around lungs
4. **Infiltration** - Abnormal infiltrate in lungs
5. **Mass** - Localized lesion/mass
6. **Nodule** - Small round lesion
7. **Pneumonia** - Lung infection
8. **Pneumothorax** - Collapsed lung (air in pleural space)
9. **Consolidation** - Lung tissue consolidation
10. **Edema** - Fluid accumulation in lungs
11. **Emphysema** - Damaged air sacs
12. **Fibrosis** - Lung tissue scarring
13. **Pleural Thickening** - Thickened pleural membrane
14. **Hernia** - Tissue protrusion

---

##  Project Structure

```
chestnut/
├── api/                          # FastAPI server implementation
│   ├── main.py                   # Main API routes
│   ├── requirements.txt           # API dependencies
│   └── frontend/
│       └── index.html            # Web interface
├── api_onnx/                     # ONNX-optimized API version
│   ├── main.py
│   ├── requirements.txt
│   └── frontend/
│       └── index.html
├── src/                          # Source code
│   ├── model.py                  # DenseNet-121 model definition
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   ├── chestxray_dataset.py      # Dataset class
│   ├── cam.py                    # CAM visualization
│   ├── cam_onnx.py               # ONNX CAM implementation
│   ├── logger.py                 # Logging utilities
│   ├── exception.py              # Custom exceptions
│   ├── checkpoint/               # Model checkpoints
│   │   ├── chexnet_full.onnx
│   │   ├── chexnet_features.onnx
│   │   └── classifier_weights.npy
│   ├── logs/                     # Training logs
│   └── results/                  # Evaluation results
├── dataset/                      # Dataset processing scripts
│   ├── batch_download_zips.py
│   ├── unzipandmovetoImages.py
│   ├── BBox_List_2017.csv
│   ├── Data_Entry_2017.csv
│   ├── Data_Entry_testing.csv
│   ├── train_val_list.txt
│   ├── test_list.txt
│   └── LongTailCXR/              # Balanced subset variants
├── notebook/                     # Jupyter notebooks
│   ├── chestnut.ipynb            # Main notebook
│   └── exploration.ipynb         # Data exploration
├── Dockerfile                    # Docker configuration
├── requirements.txt              # Project dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
```

---

##  Installation

### Prerequisites
- Python 3.8+
- PyTorch (CPU or GPU)
- CUDA 11.8+ (for GPU support)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd chestnut
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package**
   ```bash
   pip install -e .
   ```

### Docker Setup (Optional)
```bash
docker build -t chestnut .
docker run -p 8000:8000 chestnut
```

---

##  Usage

### Training

Train the model on the NIH ChestX-ray14 dataset:

```bash
python src/train.py \
  --data-dir dataset/ \
  --epochs 10 \
  --batch-size 32 \
  --device cuda  # or 'cpu'
```

**Parameters:**
- `--data-dir`: Path to dataset directory
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--device`: Device to use ('cuda' or 'cpu')

### Inference

Evaluate on test set and generate results:

```bash
python src/evaluate.py \
  --model-path src/checkpoint/chexnet_full.onnx \
  --data-dir dataset/ \
  --output-dir src/results/
```

### API Server

Start the FastAPI web server:

```bash
# Standard PyTorch version
cd api
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# Or ONNX-optimized version
cd api_onnx
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open browser to: `http://localhost:8000`

#### API Endpoints

- **POST** `/predict` - Upload X-ray and get predictions
  ```bash
  curl -X POST "http://localhost:8000/predict" \
    -F "file=@chest_xray.jpg"
  ```

- **GET** `/health` - Server health check
- **GET** `/diseases` - List all supported diseases

---

##  Dataset

### NIH ChestX-ray14
- **Images**: 112,120 frontal-view X-ray images
- **Patients**: 30,805 unique patients
- **Labels**: Multi-label annotations for 14 diseases
- **Size**: ~45GB of image data

**Dataset splits:**
- Training: 70% (78,684 images)
- Validation: 15% (16,818 images)
- Testing: 15% (16,618 images)

### Dataset Variants
- **LongTailCXR**: Balanced subset for handling class imbalance
- **PruneCXR**: MICCAI 2023 improved labels

---

## Model Architecture

### DenseNet-121
- **Base Model**: Pre-trained on ImageNet
- **Input**: 224×224 RGB images
- **Feature Extractor**: DenseNet-121 backbone
- **Classifier**: Fully connected layer (1024 → 14 classes)
- **Output**: 14 disease probabilities (sigmoid activation)

```
Input Image (224×224×3)
        ↓
   DenseNet-121 Backbone
   (Feature Extraction)
        ↓
   Fully Connected Layer
   (1024 → 14)
        ↓
   BCE with Logits Loss
   (Multi-label Classification)
```

### Training Details
- **Optimizer**: Adam (lr=0.0001)
- **Loss Function**: BCEWithLogitsLoss with class weights
- **Augmentation**: Random resizing, rotation, and normalization
- **Hardware**: GPU-accelerated training (CUDA)

---

##  Results

Model evaluation metrics on test set:

| Metric | Value |
|--------|-------|
| Average AUC-ROC | 0.87+ |
| Average Precision | 0.82+ |
| Model Size | ~30MB |
| ONNX Model Size | ~15MB |
| Inference Time (GPU) | ~50ms per image |
| Inference Time (CPU) | ~200ms per image |

Detailed results are saved in `src/results/auroc_results_*.json`

---

## Technologies Used

- **Deep Learning**: PyTorch, TorchVision
- **API Framework**: FastAPI
- **Model Optimization**: ONNX Runtime
- **Web Frontend**: HTML/CSS/JavaScript
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Pillow
- **Containerization**: Docker
- **Deployment**: Render

---

##  License

This project is based on the CheXNet paper and uses the NIH ChestX-ray14 dataset. Please cite the original papers if using this work.

---

##  Author

**Ritesh Gupta**
- Email: gupta.2002@outlook.com
- Project: [ChestNut on Render](https://chestnut-yrjc.onrender.com/)

---

##  Acknowledgments

- **CheXNet Paper**: Rajkomar et al., "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning"
- **Dataset**: NIH ChestX-ray14 dataset
- **DenseNet-121**: He et al., "Densely Connected Convolutional Networks"

---

##  Contact & Support

For issues, questions, or contributions, please reach out via email or open an issue on the repository.

---

**Last Updated**: August 2026