# Dataset Setup

This directory should contain the gender classification dataset. The dataset files are not included in the repository due to their large size.

## Required Dataset

You need to download and place the gender classification dataset in this directory. The expected structure is:

```
data/
├── FACECOM/                    # Main dataset directory (download required)
│   ├── train/                  # Training images
│   ├── val/                    # Validation images
│   └── test/                   # Test images
├── train_labels.csv            # Training labels (generated)
├── val_labels.csv              # Validation labels (generated)
├── test_labels.csv             # Test labels (generated)
└── celebrity_dataset_complete.csv  # Complete dataset metadata (generated)
```

## Dataset Setup Instructions

1. **Download the FACECOM Gender Dataset:**
   - You can find gender classification datasets on platforms like Kaggle, Google Dataset Search, or academic repositories
   - Look for datasets containing facial images with gender labels
   - Popular options include:
     - CelebA dataset
     - FACECOM dataset
     - Gender Classification datasets on Kaggle

2. **Extract the dataset:**
   - Extract the downloaded dataset to the `data/FACECOM/` directory
   - Ensure the directory structure matches the expected format above

3. **Run the setup script:**
   ```bash
   python scripts/setup_celebrity_dataset.py
   ```
   This will generate the necessary CSV files with labels and metadata.

## Notes

- All dataset files (*.csv, FACECOM/) are ignored by git to keep the repository size manageable
- Make sure you have proper permissions and follow the dataset's license terms
- The dataset should contain images organized by gender categories or with appropriate metadata files 