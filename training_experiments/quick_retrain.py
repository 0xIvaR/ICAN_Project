"""
Quick retraining script to fix gender classification issues
Uses cleaned data and balanced training for faster improvement
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import classification_report
from collections import Counter

from config import Config
from model import create_ican_model
from dataset import FACECOMDataset, get_robust_transforms

def clean_gender_labels_simple(df):
    """Clean inconsistent gender labels using majority vote"""
    print("🧹 Cleaning gender labels...")
    
    cleaned_df = df.copy()
    person_gender_mapping = {}
    
    for person_id in df['person_id'].unique():
        person_data = df[df['person_id'] == person_id]
        gender_counts = person_data['gender'].value_counts()
        majority_gender = gender_counts.index[0]
        person_gender_mapping[person_id] = majority_gender
        print(f"Person {person_id}: {dict(gender_counts)} -> {majority_gender}")
    
    cleaned_df['gender'] = cleaned_df['person_id'].map(person_gender_mapping)
    changes = (df['gender'] != cleaned_df['gender']).sum()
    print(f"✓ Fixed {changes} inconsistent labels")
    
    return cleaned_df, person_gender_mapping

def create_weighted_loss(train_df):
    """Create weighted loss function to handle class imbalance"""
    gender_counts = train_df['gender'].value_counts()
    total_samples = len(train_df)
    
    # Calculate class weights (inverse frequency)
    female_weight = total_samples / (2 * gender_counts.get('Female', 1))
    male_weight = total_samples / (2 * gender_counts.get('Male', 1))
    
    print(f"Class weights - Female: {female_weight:.3f}, Male: {male_weight:.3f}")
    
    # Create weighted loss
    class_weights = torch.tensor([female_weight, male_weight]).to(Config.DEVICE)
    gender_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    identity_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    return gender_criterion, identity_criterion

def quick_retrain_model():
    """Quick retraining with cleaned data and balanced loss"""
    
    print("🚀 QUICK GENDER CLASSIFICATION FIX")
    print("=" * 60)
    
    # 1. Load and clean data
    train_df = pd.read_csv('data/train_labels.csv')
    val_df = pd.read_csv('data/val_labels.csv')
    
    train_df_clean, gender_mapping = clean_gender_labels_simple(train_df)
    val_df_clean, _ = clean_gender_labels_simple(val_df)
    
    # 2. Check cleaned distribution
    print(f"\nCleaned training data distribution:")
    print(train_df_clean['gender'].value_counts())
    
    # 3. Create datasets with lighter augmentation
    transform_train = get_robust_transforms('train', Config.IMG_SIZE)
    transform_val = get_robust_transforms('val', Config.IMG_SIZE)
    
    unique_identities = sorted(train_df_clean['person_id'].unique())
    identity_to_idx = {identity: idx for idx, identity in enumerate(unique_identities)}
    gender_to_idx = {'Female': 0, 'Male': 1}
    
    # Create temporary CSV files for the cleaned data
    train_csv_path = 'data/train_labels_cleaned_temp.csv'
    val_csv_path = 'data/val_labels_cleaned_temp.csv'
    train_df_clean.to_csv(train_csv_path, index=False)
    val_df_clean.to_csv(val_csv_path, index=False)
    
    train_dataset = FACECOMDataset(
        csv_file=train_csv_path,
        root_dir='data/FACECOM',
        transform=transform_train,
        mode='train'
    )
    
    val_dataset = FACECOMDataset(
        csv_file=val_csv_path,
        root_dir='data/FACECOM', 
        transform=transform_val,
        mode='val'
    )
    
    # 4. Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # 5. Load existing model and modify it
    print(f"\nLoading existing model...")
    checkpoint = torch.load('checkpoints/ican_best_model.pth', map_location=Config.DEVICE)
    
    model = create_ican_model(len(unique_identities))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(Config.DEVICE)
    
    # 6. Create weighted loss functions
    gender_criterion, identity_criterion = create_weighted_loss(train_df_clean)
    
    # 7. Use higher learning rate for gender head only
    backbone_params = list(model.backbone.parameters()) + list(model.identity_head.parameters())
    gender_params = list(model.gender_head.parameters()) + list(model.shared_fc.parameters())
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},  # Very low LR for backbone
        {'params': gender_params, 'lr': 1e-3}     # Higher LR for gender-related parts
    ], weight_decay=1e-4)
    
    # 8. Quick training - focus on gender classification
    print(f"\nStarting quick retraining (10 epochs)...")
    
    for epoch in range(10):  # Just 10 epochs for quick fix
        model.train()
        train_loss = 0.0
        train_gender_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(Config.DEVICE)
            gender_labels = batch['gender'].to(Config.DEVICE)
            identity_labels = batch['identity'].to(Config.DEVICE)
            
            optimizer.zero_grad()
            
            gender_outputs, identity_outputs = model(images)
            
            # Focus more on gender loss
            gender_loss = gender_criterion(gender_outputs, gender_labels)
            identity_loss = identity_criterion(identity_outputs, identity_labels)
            
            # Give much more weight to gender task for quick fix
            total_loss = 0.8 * gender_loss + 0.2 * identity_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Statistics
            train_loss += total_loss.item()
            _, gender_pred = torch.max(gender_outputs.data, 1)
            
            train_total += gender_labels.size(0)
            train_gender_correct += (gender_pred == gender_labels).sum().item()
            
        # Validation
        model.eval()
        val_gender_correct = 0
        val_total = 0
        all_gender_preds = []
        all_gender_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(Config.DEVICE)
                gender_labels = batch['gender'].to(Config.DEVICE)
                
                gender_outputs, _ = model(images)
                _, gender_pred = torch.max(gender_outputs.data, 1)
                
                val_total += gender_labels.size(0)
                val_gender_correct += (gender_pred == gender_labels).sum().item()
                
                all_gender_preds.extend(gender_pred.cpu().numpy())
                all_gender_labels.extend(gender_labels.cpu().numpy())
        
        train_gender_acc = 100.0 * train_gender_correct / train_total
        val_gender_acc = 100.0 * val_gender_correct / val_total
        
        print(f'Epoch {epoch+1}: Train Gender: {train_gender_acc:.1f}%, Val Gender: {val_gender_acc:.1f}%')
        
        # Show detailed classification report
        if epoch % 3 == 0:
            print("\nDetailed Gender Classification:")
            print(classification_report(all_gender_labels, all_gender_preds, 
                                      target_names=['Female', 'Male'], zero_division=0))
    
    # 9. Save improved model
    torch.save({
        'epoch': 10,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'num_identity_classes': len(unique_identities),
            'backbone': Config.BACKBONE,
            'gender_mapping': gender_mapping
        }
    }, 'checkpoints/ican_quick_fixed.pth')
    
    print(f'\n✅ Quick retraining completed! Model saved as "ican_quick_fixed.pth"')
    return model

def test_quick_fixed_model():
    """Test the quickly fixed model"""
    print("\n🧪 TESTING QUICK-FIXED MODEL")
    print("=" * 60)
    
    # Load the quick-fixed model
    from improved_inference import ImprovedICANInference
    
    try:
        inference = ImprovedICANInference("checkpoints/ican_quick_fixed.pth")
        
        # Test images
        test_dir = "test_images"
        test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Testing {len(test_images)} images with quick-fixed model:")
        
        for img_file in test_images:
            img_path = os.path.join(test_dir, img_file)
            result = inference.predict(img_path, return_probabilities=True)
            
            print(f"\n📸 {img_file}:")
            print(f"   Gender: {result['gender']['prediction']} (confidence: {result['gender']['confidence']:.3f})")
            print(f"   Probabilities - Female: {result['gender']['probabilities']['Female']:.3f}, Male: {result['gender']['probabilities']['Male']:.3f}")
            
    except Exception as e:
        print(f"Error testing quick-fixed model: {e}")

if __name__ == "__main__":
    # Run quick retraining
    model = quick_retrain_model()
    
    # Test the results
    test_quick_fixed_model() 