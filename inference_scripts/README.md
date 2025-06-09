# Inference Scripts

This directory contains alternative inference implementations with different optimizations and features.

## Scripts Overview

- **`final_improved_inference.py`** - Latest and most optimized inference implementation with enhanced features
- **`improved_inference.py`** - Earlier improved version with performance optimizations

## Features

### final_improved_inference.py
- Enhanced preprocessing pipeline
- Optimized batch processing
- Better error handling
- Advanced face detection integration
- Performance monitoring
- Memory optimization

### improved_inference.py
- Basic performance improvements over original
- Simplified interface
- Basic batch processing

## Usage

For most use cases, use the main `inference.py` script in the root directory. These scripts are provided for:
- Performance comparisons
- Feature testing
- Alternative implementations

### Example Usage
```bash
# Using the final improved version
python inference_scripts/final_improved_inference.py --image test_images/test_face_0.jpg

# Using the improved version
python inference_scripts/improved_inference.py --image test_images/test_face_0.jpg --batch_size 32
```

## Notes

- These scripts may have different dependencies or configurations
- Performance characteristics may vary between versions
- Check individual script documentation for specific features and requirements
- For production use, prefer the main `inference.py` script unless you need specific features from these versions 