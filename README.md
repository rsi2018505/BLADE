# BLADE

**Banana Leaf Disease Analysis and Dataset Enhancement**

A Python-based preprocessing toolkit for banana leaf disease datasets that automates image patch creation, image resizing, and feature extraction. BLADE is designed to prepare high-quality datasets for deep learning, computer vision, and agricultural AI applications.

---

## Overview

BLADE provides a streamlined preprocessing pipeline for high-resolution banana leaf disease images. The toolkit transforms raw images into machine-learning-ready data by generating localized patches, standardizing image dimensions, and extracting informative features for downstream analysis.

This preprocessing workflow supports the development of robust disease classification, detection, localization, and explainable AI models in precision agriculture.

---

## Key Features

* Automated image patch generation from high-resolution images
* Image resizing and standardization
* Feature extraction for machine learning and statistical analysis
* Batch processing of large image datasets
* Organized preprocessing workflow
* Compatible with deep learning frameworks
* Supports agricultural AI and plant disease research

---

## Workflow

The BLADE preprocessing pipeline follows a sequential workflow:

### 1. Patch Creation

High-resolution banana leaf images are divided into smaller localized patches. This enables fine-grained disease analysis and increases the number of training samples available for machine learning models.

### 2. Image Resizing

The generated image patches are resized to standardized dimensions, ensuring consistency across the dataset and improving computational efficiency during model training.

### 3. Feature Extraction

Features are extracted from the resized patches to capture important visual characteristics such as texture, shape, and intensity patterns. These features can be used for machine learning, statistical analysis, and explainable AI studies.

### Processing Pipeline

```text
Original High-Resolution Images
                │
                ▼
        Patch Creation
                │
                ▼
         Image Resizing
                │
                ▼
       Feature Extraction
                │
                ▼
      ML/DL Ready Dataset
```

---

## Applications

* Banana Leaf Disease Classification
* Plant Disease Detection
* Deep Learning Model Training
* Computer Vision Research
* Feature Engineering and Analysis
* Agricultural AI Systems
* Explainable Artificial Intelligence (XAI)
* Precision Agriculture

---

## Project Structure

```text
BLADE/
│
├── Image Patch Creation Code.py
├── Image Resizer Code.py
├── Image Feature Extraction Code.py
├── LICENSE
└── README.md
```

---

## Requirements

Install the required Python packages:

```bash
pip install numpy pandas opencv-python matplotlib scikit-image tqdm
```

---

## Usage

### Step 1: Create Image Patches

Generate localized patches from original high-resolution banana leaf images.

```bash
python "Image Patch Creation Code.py"
```

### Step 2: Resize Generated Patches

Resize the generated patches to standardized dimensions suitable for machine learning models.

```bash
python "Image Resizer Code.py"
```

### Step 3: Extract Features

Extract image features from the resized patches.

```bash
python "Image Feature Extraction Code.py"
```

---

## Dataset Preparation Pipeline

The recommended processing order is:

```text
Original Images
      │
      ▼
Patch Generation
      │
      ▼
Image Resizing
      │
      ▼
Feature Extraction
      │
      ▼
Model Training & Analysis
```

---

## Research Applications

BLADE can be integrated into:

* Convolutional Neural Networks (CNNs)
* Vision Transformers (ViTs)
* Transfer Learning Pipelines
* Machine Learning Classification Models
* Explainable AI Frameworks
* Agricultural Disease Monitoring Systems

---

## Future Enhancements

* Multi-scale patch extraction
* Automated patch quality assessment
* Advanced texture and color feature extraction
* Parallel processing support
* Dataset statistics and visualization
* Annotation-aware patch generation
* Integration with deep learning training pipelines



---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Citation

If you use BLADE in your research, please cite the corresponding dataset or research publication associated with this project.
