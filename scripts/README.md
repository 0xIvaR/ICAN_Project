# Utility Scripts

This directory contains utility scripts for setup, testing, and analysis.

## Scripts Overview

### Setup and Installation
- **`setup_celebrity_dataset.py`** - Sets up the gender classification dataset, creates train/val/test splits and generates CSV label files
- **`check_torch.py`** - Quick check to verify PyTorch installation and CUDA availability
- **`test_installation.py`** - Comprehensive installation verification script

### Analysis and Testing
- **`analyze_data_bias.py`** - Analyzes the dataset for potential biases and class imbalances
- **`test_model_performance.py`** - Tests trained model performance with detailed metrics

## Usage

### First-time Setup
1. **Check installation:**
   ```bash
   python scripts/check_torch.py
   python scripts/test_installation.py
   ```

2. **Setup dataset:**
   ```bash
   # First, download and extract your gender classification dataset to data/FACECOM/
   python scripts/setup_celebrity_dataset.py
   ```

### Analysis
```bash
# Analyze dataset for biases
python scripts/analyze_data_bias.py

# Test model performance
python scripts/test_model_performance.py --checkpoint checkpoints/ican_best_model.pth
```

## Notes

- Run these scripts from the project root directory
- Ensure you have the required dependencies installed (`pip install -r requirements.txt`)
- Some scripts require the dataset to be properly set up first 