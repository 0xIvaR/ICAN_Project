# ICAN - Intelligent Condition-Adaptive Network

**Robust Face Recognition and Gender Classification under Adverse Visual Conditions**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

ICAN (Intelligent Condition-Adaptive Network) is a state-of-the-art deep learning model designed for robust face recognition and gender classification that works effectively even under adverse visual conditions such as:

- **Motion blur**
- **Overexposure/sunny conditions**
- **Fog**
- **Rain**
- **Low light**
- **Uneven lighting/glare**

### Key Features

- **Multi-task Learning**: Simultaneously performs gender classification and face recognition
- **Condition-Adaptive Architecture**: Specialized modules to handle adverse visual conditions
- **Attention Mechanisms**: Focuses on important facial features despite degradation
- **Robust Augmentations**: Extensive data augmentation pipeline simulating real-world conditions
- **Pre-trained Backbones**: Supports multiple backbone architectures (EfficientNet, ResNet, etc.)
- **Comprehensive Evaluation**: Detailed metrics and evaluation framework

## Architecture

ICAN uses a shared backbone with task-specific heads:

```
Input Image (224x224)
    ↓
Backbone (EfficientNet-B3)
    ↓
Attention Module
    ↓
Condition-Adaptive Blocks
    ↓
Shared Feature Extractor (512D)
    ↓
┌─────────────────┬─────────────────┐
Gender Head       Identity Head
(Binary)          (Multi-class)
```

### Model Components

1. **Backbone**: Pre-trained CNN (EfficientNet-B3 by default)
2. **Attention Module**: Spatial attention for feature enhancement
3. **Condition-Adaptive Blocks**: Residual blocks with layer normalization
4. **Multi-task Heads**: Separate heads for gender and identity classification

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/ican.git
cd ican
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python scripts/check_torch.py
python scripts/test_installation.py
```

## Dataset Setup

**⚠️ Important: Download Dataset Required**

The dataset is not included in this repository due to size constraints. You need to download and set up the gender classification dataset manually.

### Dataset Setup Instructions

1. **Download a gender classification dataset** from one of these sources:
   - [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
   - [FACECOM Dataset](https://www.kaggle.com/datasets/ashwingupta3012/human-faces) (Kaggle)
   - [Gender Classification Dataset](https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset) (Kaggle)

2. **Extract to the data directory**:
```bash
# Your dataset structure should look like:
data/
├── FACECOM/                    # Main dataset directory (download required)
│   ├── train/                  # Training images
│   ├── val/                    # Validation images  
│   └── test/                   # Test images
└── README.md                   # Setup instructions
```

3. **Run the dataset setup script**:
```bash
python scripts/setup_celebrity_dataset.py
```

This will generate the necessary CSV files (`train_labels.csv`, `val_labels.csv`, `test_labels.csv`) with proper labels and metadata.

### CSV Format (Auto-generated)

The setup script creates CSV files with this format:
```csv
image_path,gender,person_id
person_0_img_0.jpg,Male,person_0
person_0_img_1.jpg,Male,person_0
person_1_img_0.jpg,Female,person_1
...
```

### Data Splits

- **Training**: 70% of the data
- **Validation**: 15% of the data  
- **Test**: 15% of the data

## Usage

### Training

**Basic training**:
```bash
python main.py
```

**Custom training options**:
```bash
python main.py --epochs 50 --batch_size 16 --lr 0.001 --backbone efficientnet_b0
```

**Resume from checkpoint**:
```bash
python main.py --resume checkpoints/ican_best_model.pth
```

### Evaluation

**Evaluate trained model**:
```bash
python main.py --evaluate_only --resume checkpoints/ican_best_model.pth
```

### Inference

**Single image prediction**:
```bash
python inference.py --image path/to/image.jpg --checkpoint checkpoints/ican_best_model.pth
```

**Face comparison**:
```bash
python inference.py --image image1.jpg --compare image2.jpg --checkpoint checkpoints/ican_best_model.pth
```

**Programmatic usage**:
```python
from inference import load_inference_model

# Load model
model = load_inference_model("checkpoints/ican_best_model.pth")

# Make prediction
result = model.predict("path/to/image.jpg", return_probabilities=True)
print(f"Gender: {result['gender']['prediction']}")
print(f"Identity: {result['identity']['prediction']}")

# Compare faces
comparison = model.compare_faces("image1.jpg", "image2.jpg")
print(f"Same person: {comparison['is_same_person']}")
```

## Configuration

Key configuration options in `src/config.py`:

```python
# Model settings
BACKBONE = "efficientnet_b3"  # Backbone architecture
IMG_SIZE = 224               # Input image size
BATCH_SIZE = 32              # Training batch size

# Training settings
NUM_EPOCHS = 100             # Maximum epochs
LEARNING_RATE = 1e-3         # Initial learning rate
PATIENCE = 15                # Early stopping patience

# Task weights
GENDER_WEIGHT = 0.3          # Gender task weight
IDENTITY_WEIGHT = 0.7        # Identity task weight
```

## Model Performance

### Evaluation Metrics

**Gender Classification (Task A)**:
- Accuracy
- Precision  
- Recall
- F1-Score

**Face Recognition (Task B)**:
- Top-1 Accuracy
- Top-k Accuracy (k=1,3,5)
- Macro F1-Score

**Final Score**: `0.3 × (Task A Score) + 0.7 × (Task B Score)`

### Robustness Analysis

The model is evaluated under various adverse conditions:

| Condition | Impact | Mitigation Strategy |
|-----------|--------|-------------------|
| Motion Blur | High | Deblurring augmentation, attention mechanisms |
| Low Light | Medium | Brightness/contrast augmentation, normalization |
| Fog/Rain | Medium | Weather simulation, robust features |
| Overexposure | Low | Tone curve augmentation, CLAHE |

## File Structure

```
ican/
├── src/                          # Core project modules
│   ├── __init__.py
│   ├── config.py                 # Configuration settings
│   ├── dataset.py                # Dataset and data loading
│   ├── model.py                  # ICAN model architecture
│   └── trainer.py                # Training loop and metrics
├── scripts/                      # Utility scripts
│   ├── __init__.py
│   ├── setup_celebrity_dataset.py   # Dataset setup script
│   ├── check_torch.py               # PyTorch installation check
│   ├── test_installation.py         # Installation verification
│   ├── analyze_data_bias.py         # Data analysis tools
│   └── test_model_performance.py    # Model testing utilities
├── training_experiments/         # Training experiment scripts
│   ├── train_with_celebrity_data.py
│   ├── advanced_gender_fix.py
│   ├── gender_classification_fix.py
│   ├── quick_retrain.py
│   └── improved_training.py
├── inference_scripts/            # Inference variations
│   ├── final_improved_inference.py
│   └── improved_inference.py
├── test_images/                  # Test images directory
│   ├── README.md                 # Test images guide
│   ├── test_face_*.jpg          # Sample test images (included)
│   └── my_photo*.jpg            # Personal images (git ignored)
├── data/                         # Dataset directory (git ignored)
│   ├── README.md                 # Dataset setup guide
│   ├── FACECOM/                  # Main dataset (download required)
│   └── *.csv                     # Generated label files
├── archive/                      # Original files (git ignored)
├── checkpoints/                  # Model checkpoints (git ignored)
├── logs/                         # Training logs (git ignored)
├── results/                      # Training results (git ignored)
├── main.py                       # Main training script
├── inference.py                  # Main inference script
├── demo.py                       # Demo application
├── requirements.txt              # Dependencies
├── .gitignore                    # Git ignore rules
├── README.md                     # This file
└── GENDER_CLASSIFICATION_ANALYSIS_REPORT.md  # Analysis report
```

## Advanced Features

### Custom Backbones

Supported backbone architectures:
- EfficientNet (B0-B7)
- ResNet (18, 34, 50, 101, 152)
- Vision Transformer (ViT)
- RegNet
- MobileNet V3

```python
# Change backbone in src/config.py
BACKBONE = "resnet50"  # or "vit_base_patch16_224", etc.
```

### Augmentation Pipeline

Robust augmentation pipeline simulating adverse conditions:

```python
# Motion blur
A.MotionBlur(blur_limit=7, p=0.3)

# Weather conditions  
A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.3)
A.RandomRain(slant_lower=-10, slant_upper=10, p=0.3)

# Lighting conditions
A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4)
A.RandomGamma(gamma_limit=(50, 150), p=0.3)

# Noise and artifacts
A.GaussNoise(var_limit=(10.0, 50.0), p=0.3)
A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3)
```

### Multi-GPU Training

```bash
# Use DataParallel for multi-GPU training
export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py --batch_size 128
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce batch size: `--batch_size 16`
   - Use gradient accumulation
   - Use mixed precision training

2. **Dataset not found**:
   - Ensure correct directory structure
   - Check CSV file paths
   - Verify image file extensions

3. **Poor performance**:
   - Increase training epochs
   - Adjust learning rate
   - Check data quality and labels

### Performance Optimization

1. **Memory optimization**:
```python
# Enable mixed precision
torch.backends.cudnn.benchmark = True
```

2. **Data loading optimization**:
```python
# Increase number of workers
num_workers = 8  # Adjust based on CPU cores
pin_memory = True  # For GPU training
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black .
isort .
```

## Citation

If you use ICAN in your research, please cite:

```bibtex
@article{ican2024,
    title={ICAN: Intelligent Condition-Adaptive Network for Robust Face Recognition},
    author={Your Name},
    journal={arXiv preprint},
    year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [timm](https://github.com/rwightman/pytorch-image-models) for pre-trained models
- [Albumentations](https://github.com/albumentations-team/albumentations) for data augmentation
- [PyTorch](https://pytorch.org/) deep learning framework

## Contact

For questions and support:
- **Email**: your.email@example.com
- **Issues**: [GitHub Issues](https://github.com/yourusername/ican/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ican/discussions)

---

**ICAN** - Robust face recognition for the real world 🌟 