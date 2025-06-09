"""
Dataset module for FACECOM with robust augmentations for adverse conditions
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

from config import Config

class FACECOMDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, mode='train'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (string): 'train', 'val', or 'test'
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        # Initialize label encoders
        self.gender_encoder = LabelEncoder()
        self.identity_encoder = LabelEncoder()
        
        # Fit encoders
        self.data_frame['gender_encoded'] = self.gender_encoder.fit_transform(self.data_frame['gender'])
        self.data_frame['identity_encoded'] = self.identity_encoder.fit_transform(self.data_frame['person_id'])
        
        # Store number of classes
        self.num_gender_classes = len(self.gender_encoder.classes_)
        self.num_identity_classes = len(self.identity_encoder.classes_)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx]['image_path'])
        
        # Load image
        try:
            image = cv2.imread(img_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            # Fallback to PIL if OpenCV fails
            image = Image.open(img_name).convert('RGB')
            image = np.array(image)

        # Get labels
        gender_label = self.data_frame.iloc[idx]['gender_encoded']
        identity_label = self.data_frame.iloc[idx]['identity_encoded']
        
        # Apply transforms
        if self.transform:
            if isinstance(self.transform, A.Compose):
                augmented = self.transform(image=image)
                image = augmented['image']
            else:
                image = self.transform(image)

        return {
            'image': image,
            'gender': torch.tensor(gender_label, dtype=torch.long),
            'identity': torch.tensor(identity_label, dtype=torch.long),
            'image_path': self.data_frame.iloc[idx]['image_path']
        }

def get_robust_transforms(mode='train', img_size=224):
    """
    Get transforms that simulate adverse conditions and improve robustness
    """
    if mode == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.OneOf([
                # Motion blur simulation
                A.MotionBlur(blur_limit=7, p=0.3),
                A.GaussianBlur(blur_limit=7, p=0.3),
            ], p=Config.BLUR_PROB),
            
            # Weather conditions simulation
            A.OneOf([
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.3),
                A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, p=0.3),
            ], p=0.2),
            
            # Lighting conditions
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
                A.RandomGamma(gamma_limit=(50, 150), p=0.3),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            ], p=Config.BRIGHTNESS_PROB),
            
            # Noise and artifacts
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
            ], p=Config.NOISE_PROB),
            
            # Geometric transformations
            A.OneOf([
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.HorizontalFlip(p=0.5),
            ], p=0.5),
            
            # Overexposure/underexposure simulation
            A.RandomToneCurve(scale=0.1, p=0.2),
            
            # Normalization
            A.Normalize(mean=Config.MEAN, std=Config.STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=Config.MEAN, std=Config.STD),
            ToTensorV2(),
        ])

def create_data_splits(data_dir, output_dir="data"):
    """
    Create train/val/test splits from the FACECOM dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # This is a placeholder function - in practice, you would load your actual dataset
    # For now, creating a sample structure
    print("Creating data splits...")
    print("Note: Replace this with actual FACECOM dataset loading logic")
    
    # Sample data structure (replace with actual dataset loading)
    sample_data = {
        'image_path': [f'person_{i//10}_img_{i%10}.jpg' for i in range(100)],
        'gender': ['Male' if i % 2 == 0 else 'Female' for i in range(100)],
        'person_id': [f'person_{i//10}' for i in range(100)]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Split the data
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['gender'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['gender'])
    
    # Save splits
    train_df.to_csv(os.path.join(output_dir, 'train_labels.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_labels.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_labels.csv'), index=False)
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    return train_df, val_df, test_df

def get_data_loaders(batch_size=32):
    """
    Get data loaders for train, validation, and test sets
    """
    # Create transforms
    train_transform = get_robust_transforms('train', Config.IMG_SIZE)
    val_transform = get_robust_transforms('val', Config.IMG_SIZE)
    
    # Create datasets
    train_dataset = FACECOMDataset(
        csv_file=Config.TRAIN_CSV,
        root_dir=Config.DATA_ROOT,
        transform=train_transform,
        mode='train'
    )
    
    val_dataset = FACECOMDataset(
        csv_file=Config.VAL_CSV,
        root_dir=Config.DATA_ROOT,
        transform=val_transform,
        mode='val'
    )
    
    test_dataset = FACECOMDataset(
        csv_file=Config.TEST_CSV,
        root_dir=Config.DATA_ROOT,
        transform=val_transform,
        mode='test'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.num_identity_classes 