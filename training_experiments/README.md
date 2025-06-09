# Training Experiments

This directory contains various training experiment scripts that explore different approaches and improvements for the gender classification model.

## Experiment Scripts

### Main Training Scripts
- **`train_with_celebrity_data.py`** - Training script specifically designed for celebrity datasets with enhanced preprocessing
- **`improved_training.py`** - Enhanced training pipeline with advanced augmentations and optimization techniques

### Model Fixes and Improvements
- **`advanced_gender_fix.py`** - Advanced gender classification improvements with better architecture and loss functions
- **`gender_classification_fix.py`** - Basic gender classification fixes and improvements
- **`quick_retrain.py`** - Quick retraining script for fine-tuning existing models

## Usage

These scripts are experimental versions that were used during development to test different approaches. For regular training, use the main `main.py` script in the root directory.

### Running Experiments
```bash
# Example: Run celebrity data training
python training_experiments/train_with_celebrity_data.py

# Example: Run improved training
python training_experiments/improved_training.py --epochs 50 --batch_size 16

# Example: Quick retrain existing model
python training_experiments/quick_retrain.py --checkpoint checkpoints/base_model.pth
```

## Notes

- These scripts may have different parameter configurations than the main training script
- Some scripts might require specific dataset formats or additional dependencies
- Results from experiments are typically saved to separate directories to avoid conflicts
- Check individual script documentation for specific usage instructions

## Experimental Features

The scripts in this directory test various improvements:
- Different backbone architectures
- Alternative loss functions
- Various augmentation strategies
- Different optimization approaches
- Multi-task learning variations
- Transfer learning techniques 