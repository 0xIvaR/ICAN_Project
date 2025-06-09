"""
Train ICAN model with clean celebrity dataset
Uses the properly balanced celebrity face dataset for training
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import time

from config import Config
from model import create_ican_model, ICANLoss
from dataset import FACECOMDataset, get_robust_transforms
from trainer import EarlyStopping, MetricsTracker

def train_celebrity_ican_model():
    """Train ICAN model with the celebrity dataset"""
    
    print("🎬 TRAINING ICAN WITH CELEBRITY DATASET")
    print("=" * 70)
    
    # 1. Load the clean dataset
    print("📊 Loading clean celebrity dataset...")
    train_df = pd.read_csv('data/train_labels.csv')
    val_df = pd.read_csv('data/val_labels.csv')
    
    print(f"   Training samples: {len(train_df)}")
    print(f"   Validation samples: {len(val_df)}")
    
    # Check gender distribution
    train_gender_dist = train_df['gender'].value_counts()
    val_gender_dist = val_df['gender'].value_counts()
    
    print(f"\n   Training gender distribution:")
    for gender, count in train_gender_dist.items():
        pct = (count / len(train_df)) * 100
        print(f"     {gender}: {count} ({pct:.1f}%)")
    
    print(f"   Validation gender distribution:")
    for gender, count in val_gender_dist.items():
        pct = (count / len(val_df)) * 100
        print(f"     {gender}: {count} ({pct:.1f}%)")
    
    # 2. Create datasets and data loaders
    print(f"\n🔄 Creating datasets and data loaders...")
    
    # Create transforms
    train_transform = get_robust_transforms('train', Config.IMG_SIZE)
    val_transform = get_robust_transforms('val', Config.IMG_SIZE)
    
    # Create temporary CSV files for FACECOMDataset
    train_csv_path = 'data/train_labels_temp.csv'
    val_csv_path = 'data/val_labels_temp.csv'
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    
    # Create datasets
    train_dataset = FACECOMDataset(
        csv_file=train_csv_path,
        root_dir='data/FACECOM',
        transform=train_transform,
        mode='train'
    )
    
    val_dataset = FACECOMDataset(
        csv_file=val_csv_path,
        root_dir='data/FACECOM',
        transform=val_transform,
        mode='val'
    )
    
    print(f"   ✅ Training dataset: {len(train_dataset)} samples")
    print(f"   ✅ Validation dataset: {len(val_dataset)} samples")
    print(f"   ✅ Gender classes: {train_dataset.num_gender_classes}")
    print(f"   ✅ Identity classes: {train_dataset.num_identity_classes}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 3. Create model
    print(f"\n🏗️  Creating ICAN model...")
    num_identity_classes = train_dataset.num_identity_classes
    model = create_ican_model(num_identity_classes)
    model.to(Config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   ✅ Model: ICAN with {Config.BACKBONE} backbone")
    print(f"   ✅ Identity classes: {num_identity_classes}")
    print(f"   ✅ Total parameters: {total_params:,}")
    print(f"   ✅ Trainable parameters: {trainable_params:,}")
    print(f"   ✅ Device: {Config.DEVICE}")
    
    # 4. Setup loss function (balanced for clean data)
    print(f"\n⚖️  Setting up balanced loss function...")
    criterion = ICANLoss(
        gender_weight=0.5,    # Equal weight since data is balanced
        identity_weight=0.5,
        label_smoothing=0.1
    )
    
    # 5. Setup optimizer with different learning rates
    print(f"\n🎯 Setting up optimizer...")
    backbone_params = list(model.backbone.parameters())
    head_params = (list(model.gender_head.parameters()) + 
                   list(model.identity_head.parameters()) + 
                   list(model.shared_fc.parameters()))
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': Config.LEARNING_RATE * 0.1},  # Lower LR for backbone
        {'params': head_params, 'lr': Config.LEARNING_RATE}  # Higher LR for heads
    ], weight_decay=Config.WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    # 6. Setup training utilities
    print(f"\n🛠️  Setting up training utilities...")
    early_stopping = EarlyStopping(patience=Config.PATIENCE, min_delta=0.001)
    metrics_tracker = MetricsTracker()
    
    best_val_accuracy = 0.0
    training_start_time = time.time()
    
    print(f"   ✅ Early stopping patience: {Config.PATIENCE}")
    print(f"   ✅ Max epochs: {Config.NUM_EPOCHS}")
    
    # 7. Training loop
    print(f"\n🚀 STARTING TRAINING")
    print("=" * 70)
    
    for epoch in range(Config.NUM_EPOCHS):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_gender_correct = 0
        train_identity_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(Config.DEVICE)
            gender_labels = batch['gender'].to(Config.DEVICE)
            identity_labels = batch['identity'].to(Config.DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            gender_outputs, identity_outputs = model(images)
            loss, gender_loss, identity_loss = criterion(
                gender_outputs, identity_outputs, gender_labels, identity_labels
            )
            
            # Backward pass
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
            
            # Print progress every 50 batches
            if batch_idx % 50 == 0:
                batch_acc = 100.0 * (gender_pred == gender_labels).sum().item() / gender_labels.size(0)
                print(f'   Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Gender Acc: {batch_acc:.1f}%')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_gender_correct = 0
        val_identity_correct = 0
        val_total = 0
        
        all_gender_preds = []
        all_gender_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(Config.DEVICE)
                gender_labels = batch['gender'].to(Config.DEVICE)
                identity_labels = batch['identity'].to(Config.DEVICE)
                
                # Forward pass
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
        
        # Combined accuracy (equal weight since data is balanced)
        combined_val_acc = 0.5 * val_gender_acc + 0.5 * val_identity_acc
        
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch results
        print(f'\n📊 EPOCH {epoch+1} RESULTS (Time: {epoch_time:.1f}s):')
        print(f'   Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'   Train - Gender: {train_gender_acc:.2f}%, Identity: {train_identity_acc:.2f}%')
        print(f'   Val   - Gender: {val_gender_acc:.2f}%, Identity: {val_identity_acc:.2f}%')
        print(f'   Combined Val Accuracy: {combined_val_acc:.2f}%')
        print(f'   Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Detailed gender classification report every 5 epochs
        if epoch % 5 == 0:
            print(f"\n   Gender Classification Report:")
            report = classification_report(all_gender_labels, all_gender_preds, 
                                         target_names=['Female', 'Male'], zero_division=0)
            print(f"   {report}")
        
        # Save best model
        if combined_val_acc > best_val_accuracy:
            best_val_accuracy = combined_val_acc
            
            # Save model checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_accuracy': best_val_accuracy,
                'train_gender_acc': train_gender_acc,
                'val_gender_acc': val_gender_acc,
                'config': {
                    'num_identity_classes': num_identity_classes,
                    'backbone': Config.BACKBONE,
                    'dataset': 'celebrity_faces',
                    'total_samples': len(train_dataset) + len(val_dataset)
                }
            }, 'checkpoints/ican_celebrity_best.pth')
            
            print(f'   🎉 NEW BEST MODEL! Accuracy: {best_val_accuracy:.2f}%')
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping check
        early_stopping(val_loss / len(val_loader))
        if early_stopping.should_stop:
            print(f'\n⏹️  Early stopping triggered at epoch {epoch+1}')
            break
        
        print("-" * 70)
    
    # Training completed
    total_training_time = time.time() - training_start_time
    print(f'\n🎉 TRAINING COMPLETED!')
    print("=" * 70)
    print(f'   Total training time: {total_training_time/3600:.1f} hours')
    print(f'   Best validation accuracy: {best_val_accuracy:.2f}%')
    print(f'   Final gender accuracy: {val_gender_acc:.2f}%')
    print(f'   Model saved as: checkpoints/ican_celebrity_best.pth')
    
    # Clean up temporary files
    if os.path.exists(train_csv_path):
        os.remove(train_csv_path)
    if os.path.exists(val_csv_path):
        os.remove(val_csv_path)
    
    return model, best_val_accuracy

def test_celebrity_model():
    """Test the trained celebrity model"""
    
    print(f"\n🧪 TESTING CELEBRITY MODEL")
    print("=" * 70)
    
    try:
        # Test on the test images from our test_images directory
        from final_improved_inference import FinalImprovedInference
        
        # Try to load the new celebrity model
        celebrity_model_path = "checkpoints/ican_celebrity_best.pth"
        if os.path.exists(celebrity_model_path):
            print(f"🎯 Testing with celebrity-trained model...")
            inference = FinalImprovedInference(celebrity_model_path)
        else:
            print(f"⚠️  Celebrity model not found, using original model...")
            inference = FinalImprovedInference("checkpoints/ican_best_model.pth")
        
        # Test on our original test images
        test_dir = "test_images"
        if os.path.exists(test_dir):
            test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"Testing on {len(test_images)} test images:")
            
            male_predictions = 0
            female_predictions = 0
            
            for img_file in test_images:
                img_path = os.path.join(test_dir, img_file)
                result = inference.predict_with_confidence_analysis(img_path)
                
                print(f"\n📸 {img_file}:")
                print(f"   Prediction: {result['gender']['prediction']} (confidence: {result['gender']['confidence']:.3f})")
                print(f"   Probabilities - Female: {result['gender']['probabilities']['Female']:.3f}, Male: {result['gender']['probabilities']['Male']:.3f}")
                
                if result['gender']['prediction'] == 'Male':
                    male_predictions += 1
                else:
                    female_predictions += 1
            
            print(f"\n📊 TEST RESULTS SUMMARY:")
            print(f"   Male predictions: {male_predictions}/{len(test_images)} ({male_predictions/len(test_images)*100:.1f}%)")
            print(f"   Female predictions: {female_predictions}/{len(test_images)} ({female_predictions/len(test_images)*100:.1f}%)")
            
            if male_predictions > 0:
                print(f"   🎉 SUCCESS! Model can now predict males!")
            else:
                print(f"   ⚠️  Still predicting all as female...")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")

if __name__ == "__main__":
    # Train the model
    model, best_accuracy = train_celebrity_ican_model()
    
    # Test the model
    test_celebrity_model()
    
    print(f"\n🚀 CELEBRITY ICAN TRAINING COMPLETE!")
    print(f"   Ready for improved gender classification!") 