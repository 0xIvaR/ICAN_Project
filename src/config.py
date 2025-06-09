"""
Configuration file for ICAN - Intelligent Condition-Adaptive Network
"""

import torch

class Config:
    # Data paths
    DATA_ROOT = "data/FACECOM"
    TRAIN_CSV = "data/train_labels.csv"
    VAL_CSV = "data/val_labels.csv"
    TEST_CSV = "data/test_labels.csv"
    
    # Model configuration
    MODEL_NAME = "ICAN"
    BACKBONE = "efficientnet_b3"  # Can be changed to resnet50, efficientnet_b0, etc.
    PRETRAINED = True
    NUM_CLASSES_GENDER = 2  # Male, Female
    NUM_CLASSES_IDENTITY = None  # Will be set dynamically based on dataset
    
    # Training configuration
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 15  # Early stopping patience
    
    # Image configuration
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Training splits
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Loss weights for multi-task learning
    GENDER_WEIGHT = 0.3
    IDENTITY_WEIGHT = 0.7
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Augmentation probabilities
    AUGMENTATION_PROB = 0.7
    BLUR_PROB = 0.3
    NOISE_PROB = 0.3
    BRIGHTNESS_PROB = 0.3
    CONTRAST_PROB = 0.3
    
    # Scheduler
    SCHEDULER_STEP_SIZE = 30
    SCHEDULER_GAMMA = 0.1
    
    # Checkpoints
    CHECKPOINT_DIR = "checkpoints"
    BEST_MODEL_PATH = "checkpoints/ican_best_model.pth"
    
    # Evaluation
    TOP_K = [1, 3, 5]  # For top-k accuracy in face recognition 