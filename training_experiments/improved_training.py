"""
Improved training script for ICAN with fixes for gender classification issues
Addresses data labeling inconsistencies and bias problems
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

from config import Config
from model import create_ican_model, ICANLoss
from dataset import ICANDataset, get_robust_transforms
from trainer import EarlyStopping, MetricsTracker

class ImprovedICANLoss(nn.Module):
    """
    Improved loss function with class balancing and focal loss for hard examples
    """
    def __init__(self, gender_weight=0.5, identity_weight=0.5, 
                 class_weights=None, focal_alpha=0.25, focal_gamma=2.0):
        super(ImprovedICANLoss, self).__init__()
        self.gender_weight = gender_weight
        self.identity_weight = identity_weight
        
        # Use class weights if provided
        if class_weights is not None:
            gender_weights = torch.tensor([class_weights['female_weight'], class_weights['male_weight']])
        else:
            gender_weights = None
        
        # Standard cross-entropy with class weights
        self.gender_criterion = nn.CrossEntropyLoss(weight=gender_weights, label_smoothing=0.1)
        self.identity_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        
        # Focal loss parameters
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def focal_loss(self, inputs, targets, alpha, gamma):
        """Focal loss to focus on hard examples"""
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def forward(self, gender_logits, identity_logits, gender_targets, identity_targets):
        # Standard loss
        gender_loss = self.gender_criterion(gender_logits, gender_targets)
        identity_loss = self.identity_criterion(identity_logits, identity_targets)
        
        # Add focal loss for gender (to focus on hard examples)
        focal_gender_loss = self.focal_loss(gender_logits, gender_targets, 
                                          self.focal_alpha, self.focal_gamma)
        
        # Combine losses
        combined_gender_loss = 0.7 * gender_loss + 0.3 * focal_gender_loss
        
        total_loss = (self.gender_weight * combined_gender_loss + 
                     self.identity_weight * identity_loss)
        
        return total_loss, combined_gender_loss, identity_loss

def clean_gender_labels(df):
    """
    Clean inconsistent gender labels by using majority vote per person
    """
    print("🧹 Cleaning gender labels...")
    
    cleaned_df = df.copy()
    
    # For each person, determine the majority gender
    person_gender_mapping = {}
    
    for person_id in df['person_id'].unique():
        person_data = df[df['person_id'] == person_id]
        gender_counts = person_data['gender'].value_counts()
        
        # Use majority vote
        majority_gender = gender_counts.index[0]
        person_gender_mapping[person_id] = majority_gender
        
        print(f"Person {person_id}: {dict(gender_counts)} -> {majority_gender}")
    
    # Apply consistent labels
    cleaned_df['gender'] = cleaned_df['person_id'].map(person_gender_mapping)
    
    # Check how many labels were changed
    changes = (df['gender'] != cleaned_df['gender']).sum()
    print(f"✓ Fixed {changes} inconsistent labels")
    
    return cleaned_df, person_gender_mapping

def create_balanced_sampler(dataset, gender_column):
    """Create a weighted sampler to balance gender classes during training"""
    
    # Count gender distribution
    gender_counts = Counter(gender_column)
    total_samples = len(gender_column)
    
    # Calculate weights (inverse frequency)
    class_weights = {}
    for gender, count in gender_counts.items():
        class_weights[gender] = total_samples / (len(gender_counts) * count)
    
    # Create sample weights
    sample_weights = []
    for gender in gender_column:
        sample_weights.append(class_weights[gender])
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    print(f"Created balanced sampler with weights: {class_weights}")
    return sampler

def retrain_improved_model():
    """Retrain the model with improved techniques"""
    
    print("🚀 IMPROVED ICAN TRAINING")
    print("=" * 60)
    
    # 1. Load and clean data
    print("\n1. Loading and cleaning training data...")
    train_df = pd.read_csv('data/train_labels.csv')
    val_df = pd.read_csv('data/val_labels.csv')
    
    # Clean inconsistent labels
    train_df_clean, gender_mapping = clean_gender_labels(train_df)
    val_df_clean, _ = clean_gender_labels(val_df)
    
    # Save cleaned data
    train_df_clean.to_csv('data/train_labels_cleaned.csv', index=False)
    val_df_clean.to_csv('data/val_labels_cleaned.csv', index=False)
    
    # 2. Create datasets
    print("\n2. Creating datasets...")
    transform_train = get_robust_transforms('train', Config.IMG_SIZE)
    transform_val = get_robust_transforms('val', Config.IMG_SIZE)
    
    # Get unique identities and create label encoders
    unique_identities = sorted(train_df_clean['person_id'].unique())
    identity_to_idx = {identity: idx for idx, identity in enumerate(unique_identities)}
    gender_to_idx = {'Female': 0, 'Male': 1}
    
    # Create datasets
    train_dataset = ICANDataset(
        train_df_clean, 
        root_dir='data/FACECOM',
        transform=transform_train,
        identity_to_idx=identity_to_idx,
        gender_to_idx=gender_to_idx
    )
    
    val_dataset = ICANDataset(
        val_df_clean,
        root_dir='data/FACECOM', 
        transform=transform_val,
        identity_to_idx=identity_to_idx,
        gender_to_idx=gender_to_idx
    )
    
    # 3. Create balanced data loader
    print("\n3. Creating balanced data loaders...")
    train_sampler = create_balanced_sampler(train_dataset, train_df_clean['gender'].values)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # 4. Create improved model
    print("\n4. Creating improved model...")
    num_identity_classes = len(unique_identities)
    model = create_ican_model(num_identity_classes)
    model.to(Config.DEVICE)
    
    # 5. Calculate class weights
    gender_counts = train_df_clean['gender'].value_counts()
    total_samples = len(train_df_clean)
    class_weights = {
        'female_weight': total_samples / (2 * gender_counts.get('Female', 1)),
        'male_weight': total_samples / (2 * gender_counts.get('Male', 1))
    }
    
    # 6. Create improved loss function
    criterion = ImprovedICANLoss(
        gender_weight=0.6,  # Give more weight to gender task
        identity_weight=0.4,
        class_weights=class_weights
    )
    
    # 7. Setup optimizer with different learning rates
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.gender_head.parameters()) + list(model.identity_head.parameters())
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': Config.LEARNING_RATE * 0.1},  # Lower LR for backbone
        {'params': head_params, 'lr': Config.LEARNING_RATE}  # Higher LR for heads
    ], weight_decay=Config.WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # 8. Training loop
    print("\n5. Starting improved training...")
    early_stopping = EarlyStopping(patience=Config.PATIENCE, min_delta=0.001)
    metrics_tracker = MetricsTracker()
    
    best_val_accuracy = 0.0
    
    for epoch in range(Config.NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_gender_correct = 0
        train_identity_correct = 0
        train_total = 0
        
        for batch_idx, (images, gender_labels, identity_labels) in enumerate(train_loader):
            images = images.to(Config.DEVICE)
            gender_labels = gender_labels.to(Config.DEVICE)
            identity_labels = identity_labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            
            gender_outputs, identity_outputs = model(images)
            loss, gender_loss, identity_loss = criterion(
                gender_outputs, identity_outputs, gender_labels, identity_labels
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, gender_pred = torch.max(gender_outputs.data, 1)
            _, identity_pred = torch.max(identity_outputs.data, 1)
            
            train_total += gender_labels.size(0)
            train_gender_correct += (gender_pred == gender_labels).sum().item()
            train_identity_correct += (identity_pred == identity_labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_gender_correct = 0
        val_identity_correct = 0
        val_total = 0
        
        all_gender_preds = []
        all_gender_labels = []
        
        with torch.no_grad():
            for images, gender_labels, identity_labels in val_loader:
                images = images.to(Config.DEVICE)
                gender_labels = gender_labels.to(Config.DEVICE) 
                identity_labels = identity_labels.to(Config.DEVICE)
                
                gender_outputs, identity_outputs = model(images)
                loss, _, _ = criterion(gender_outputs, identity_outputs, gender_labels, identity_labels)
                
                val_loss += loss.item()
                _, gender_pred = torch.max(gender_outputs.data, 1)
                _, identity_pred = torch.max(identity_outputs.data, 1)
                
                val_total += gender_labels.size(0)
                val_gender_correct += (gender_pred == gender_labels).sum().item()
                val_identity_correct += (identity_pred == identity_labels).sum().item()
                
                # Collect for detailed metrics
                all_gender_preds.extend(gender_pred.cpu().numpy())
                all_gender_labels.extend(gender_labels.cpu().numpy())
        
        # Calculate metrics
        train_gender_acc = 100.0 * train_gender_correct / train_total
        train_identity_acc = 100.0 * train_identity_correct / train_total
        val_gender_acc = 100.0 * val_gender_correct / val_total
        val_identity_acc = 100.0 * val_identity_correct / val_total
        
        combined_val_acc = 0.6 * val_gender_acc + 0.4 * val_identity_acc
        
        print(f'\nEpoch {epoch+1}:')
        print(f'Train - Gender: {train_gender_acc:.2f}%, Identity: {train_identity_acc:.2f}%')
        print(f'Val   - Gender: {val_gender_acc:.2f}%, Identity: {val_identity_acc:.2f}%')
        print(f'Combined Val Accuracy: {combined_val_acc:.2f}%')
        
        # Detailed gender classification report
        if epoch % 5 == 0:
            gender_names = ['Female', 'Male']
            print("\nGender Classification Report:")
            print(classification_report(all_gender_labels, all_gender_preds, 
                                      target_names=gender_names, zero_division=0))
        
        # Save best model
        if combined_val_acc > best_val_accuracy:
            best_val_accuracy = combined_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_accuracy': best_val_accuracy,
                'config': {
                    'num_identity_classes': num_identity_classes,
                    'backbone': Config.BACKBONE,
                    'class_weights': class_weights,
                    'gender_mapping': gender_mapping
                }
            }, 'checkpoints/ican_improved_model.pth')
            print(f'New best model saved! Accuracy: {best_val_accuracy:.2f}%')
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping
        early_stopping(val_loss / len(val_loader))
        if early_stopping.should_stop:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break
    
    print(f'\n✅ Training completed! Best validation accuracy: {best_val_accuracy:.2f}%')
    return model, best_val_accuracy

if __name__ == "__main__":
    retrain_improved_model() 