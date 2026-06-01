# BLADE

**Banana Leaf Disease Analysis and Dataset Enhancement**

A Python-based preprocessing toolkit for banana leaf disease datasets that automates image resizing, patch generation, and feature extraction. Designed to prepare high-quality datasets for deep learning, computer vision, and agricultural AI applications.

---

## Overview

BLADE provides a streamlined preprocessing pipeline for banana leaf disease image datasets. It enables researchers and developers to transform raw high-resolution images into standardized, machine-learning-ready data through automated resizing, patch extraction, and feature generation.

The toolkit is particularly useful for developing robust disease classification, detection, and localization models using deep learning frameworks.


## Key Features

* Image resizing and standardization
* High-resolution image patch generation
* Automated feature extraction
* Batch processing support
* Dataset organization and preparation
* Compatible with deep learning workflows
* Suitable for agricultural image analytics and plant disease research


## Applications

* Banana leaf disease classification
* Deep learning model training
* Computer vision research
* Feature engineering and analysis
* Dataset preprocessing and augmentation
* Agricultural AI systems



## Project Structure

```text
BLADE/
│
├── Image_Resizer_Code.ipynb
├── Image_Patch_Creation_Code.ipynb
├── Image_Feature_Extraction_Code.ipynb
├── datasets/
├── outputs/
└── README.md
```


## Workflow

### 1. Image Resizing

Standardizes image dimensions to ensure consistency across the dataset.

### 2. Patch Generation

Extracts localized image patches from high-resolution banana leaf images for fine-grained disease analysis.

### 3. Feature Extraction

Generates meaningful image descriptors that can be used for machine learning and statistical analysis.



## Requirements

```bash
pip install numpy pandas opencv-python matplotlib scikit-image tqdm
```


## Use Cases

* Disease diagnosis using AI
* Precision agriculture
* Plant pathology research
* Dataset preparation for CNNs and Vision Transformers
* Explainable AI studies in agriculture


## Future Enhancements

* Multi-scale patch extraction
* Advanced texture and color descriptors
* Automated quality assessment
* Parallelized processing pipeline
* Integration with deep learning training frameworks


## License

This project is intended for academic, educational, and research purposes.
