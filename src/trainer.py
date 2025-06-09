"""
Training module for ICAN model
Includes training loop, validation, metrics tracking, and model checkpointing
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

from config import Config
from model import ICAN, ICANLoss

class MetricsTracker:
    """Track and compute metrics for both tasks"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.gender_preds = []
        self.gender_targets = []
        self.identity_preds = []
        self.identity_targets = []
        self.losses = []
        self.gender_losses = []
        self.identity_losses = []
    
    def update(self, gender_preds, identity_preds, gender_targets, identity_targets, 
               total_loss, gender_loss, identity_loss):
        """Update metrics with batch results"""
        self.gender_preds.extend(gender_preds.cpu().numpy())
        self.gender_targets.extend(gender_targets.cpu().numpy())
        self.identity_preds.extend(identity_preds.cpu().numpy())
        self.identity_targets.extend(identity_targets.cpu().numpy())
        self.losses.append(total_loss)
        self.gender_losses.append(gender_loss)
        self.identity_losses.append(identity_loss)
    
    def compute_metrics(self):
        """Compute final metrics"""
        gender_preds = np.array(self.gender_preds)
        gender_targets = np.array(self.gender_targets)
        identity_preds = np.array(self.identity_preds)
        identity_targets = np.array(self.identity_targets)
        
        # Gender classification metrics
        gender_acc = accuracy_score(gender_targets, gender_preds)
        gender_precision, gender_recall, gender_f1, _ = precision_recall_fscore_support(
            gender_targets, gender_preds, average='binary'
        )
        
        # Identity recognition metrics
        identity_acc = accuracy_score(identity_targets, identity_preds)
        identity_precision, identity_recall, identity_f1, _ = precision_recall_fscore_support(
            identity_targets, identity_preds, average='macro'
        )
        
        # Top-k accuracy for identity
        top_k_acc = {}
        for k in Config.TOP_K:
            if k <= len(np.unique(identity_targets)):
                top_k_acc[f'top_{k}'] = self._compute_top_k_accuracy(
                    identity_targets, identity_preds, k
                )
        
        # Final scores according to problem definition
        gender_score = (gender_acc + gender_f1) / 2
        identity_score = (identity_acc + identity_f1) / 2
        final_score = Config.GENDER_WEIGHT * gender_score + Config.IDENTITY_WEIGHT * identity_score
        
        return {
            'total_loss': np.mean(self.losses),
            'gender_loss': np.mean(self.gender_losses),
            'identity_loss': np.mean(self.identity_losses),
            'gender_accuracy': gender_acc,
            'gender_precision': gender_precision,
            'gender_recall': gender_recall,
            'gender_f1': gender_f1,
            'identity_accuracy': identity_acc,
            'identity_precision': identity_precision,
            'identity_recall': identity_recall,
            'identity_f1': identity_f1,
            'gender_score': gender_score,
            'identity_score': identity_score,
            'final_score': final_score,
            **top_k_acc
        }
    
    def _compute_top_k_accuracy(self, targets, preds, k):
        """Compute top-k accuracy (simplified version for this example)"""
        # In practice, you'd need the full probability distributions
        # This is a simplified version assuming we have the predictions
        return accuracy_score(targets, preds)  # Placeholder

class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience=15, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False

class ICANTrainer:
    """Main trainer class for ICAN model"""
    
    def __init__(self, model, train_loader, val_loader, test_loader=None):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Loss function
        self.criterion = ICANLoss(
            gender_weight=Config.GENDER_WEIGHT,
            identity_weight=Config.IDENTITY_WEIGHT
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        # Scheduler
        self.scheduler = StepLR(
            self.optimizer,
            step_size=Config.SCHEDULER_STEP_SIZE,
            gamma=Config.SCHEDULER_GAMMA
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=Config.PATIENCE)
        
        # Training history
        self.history = defaultdict(list)
        
        # Create checkpoint directory
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        metrics = MetricsTracker()
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            images = batch['image'].to(Config.DEVICE)
            gender_targets = batch['gender'].to(Config.DEVICE)
            identity_targets = batch['identity'].to(Config.DEVICE)
            
            # Forward pass
            gender_logits, identity_logits = self.model(images)
            
            # Compute loss
            total_loss, gender_loss, identity_loss = self.criterion(
                gender_logits, identity_logits, gender_targets, identity_targets
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Get predictions
            gender_preds = torch.argmax(gender_logits, dim=1)
            identity_preds = torch.argmax(identity_logits, dim=1)
            
            # Update metrics
            metrics.update(
                gender_preds, identity_preds, gender_targets, identity_targets,
                total_loss.item(), gender_loss.item(), identity_loss.item()
            )
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'G_Loss': f'{gender_loss.item():.4f}',
                'I_Loss': f'{identity_loss.item():.4f}'
            })
        
        return metrics.compute_metrics()
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        metrics = MetricsTracker()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch in pbar:
                images = batch['image'].to(Config.DEVICE)
                gender_targets = batch['gender'].to(Config.DEVICE)
                identity_targets = batch['identity'].to(Config.DEVICE)
                
                # Forward pass
                gender_logits, identity_logits = self.model(images)
                
                # Compute loss
                total_loss, gender_loss, identity_loss = self.criterion(
                    gender_logits, identity_logits, gender_targets, identity_targets
                )
                
                # Get predictions
                gender_preds = torch.argmax(gender_logits, dim=1)
                identity_preds = torch.argmax(identity_logits, dim=1)
                
                # Update metrics
                metrics.update(
                    gender_preds, identity_preds, gender_targets, identity_targets,
                    total_loss.item(), gender_loss.item(), identity_loss.item()
                )
                
                # Update progress bar
                pbar.set_postfix({
                    'Val_Loss': f'{total_loss.item():.4f}',
                    'G_Acc': f'{accuracy_score(gender_targets.cpu(), gender_preds.cpu()):.4f}',
                    'I_Acc': f'{accuracy_score(identity_targets.cpu(), identity_preds.cpu()):.4f}'
                })
        
        return metrics.compute_metrics()
    
    def train(self, num_epochs=None):
        """Main training loop"""
        if num_epochs is None:
            num_epochs = Config.NUM_EPOCHS
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {Config.DEVICE}")
        print(f"Model: {Config.MODEL_NAME}")
        
        best_val_score = 0.0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            self.history['train_loss'].append(train_metrics['total_loss'])
            self.history['val_loss'].append(val_metrics['total_loss'])
            self.history['train_final_score'].append(train_metrics['final_score'])
            self.history['val_final_score'].append(val_metrics['final_score'])
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch + 1} Summary:")
            print(f"Time: {epoch_time:.2f}s")
            print(f"Train Loss: {train_metrics['total_loss']:.4f} | Val Loss: {val_metrics['total_loss']:.4f}")
            print(f"Train Score: {train_metrics['final_score']:.4f} | Val Score: {val_metrics['final_score']:.4f}")
            print(f"Gender - Train Acc: {train_metrics['gender_accuracy']:.4f} | Val Acc: {val_metrics['gender_accuracy']:.4f}")
            print(f"Identity - Train Acc: {train_metrics['identity_accuracy']:.4f} | Val Acc: {val_metrics['identity_accuracy']:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_metrics['final_score'] > best_val_score:
                best_val_score = val_metrics['final_score']
                self.save_checkpoint(epoch, val_metrics, 'best')
                print(f"New best model saved! Score: {best_val_score:.4f}")
            
            # Early stopping
            if self.early_stopping(val_metrics['total_loss'], self.model):
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time / 60:.2f} minutes")
        print(f"Best validation score: {best_val_score:.4f}")
        
        return self.history
    
    def save_checkpoint(self, epoch, metrics, checkpoint_type='latest'):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': {
                'backbone': Config.BACKBONE,
                'num_identity_classes': self.model.num_identity_classes,
                'img_size': Config.IMG_SIZE,
            }
        }
        
        if checkpoint_type == 'best':
            path = Config.BEST_MODEL_PATH
        else:
            path = os.path.join(Config.CHECKPOINT_DIR, f'ican_epoch_{epoch}.pth')
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Epoch: {checkpoint['epoch']}")
        print(f"Validation Score: {checkpoint['metrics']['final_score']:.4f}")
        
        return checkpoint
    
    def evaluate(self, data_loader=None):
        """Evaluate model on test set"""
        if data_loader is None:
            data_loader = self.test_loader
        
        if data_loader is None:
            print("No test data loader provided")
            return None
        
        print("Evaluating on test set...")
        self.model.eval()
        metrics = MetricsTracker()
        
        with torch.no_grad():
            pbar = tqdm(data_loader, desc="Testing")
            for batch in pbar:
                images = batch['image'].to(Config.DEVICE)
                gender_targets = batch['gender'].to(Config.DEVICE)
                identity_targets = batch['identity'].to(Config.DEVICE)
                
                # Forward pass
                gender_logits, identity_logits = self.model(images)
                
                # Compute loss
                total_loss, gender_loss, identity_loss = self.criterion(
                    gender_logits, identity_logits, gender_targets, identity_targets
                )
                
                # Get predictions
                gender_preds = torch.argmax(gender_logits, dim=1)
                identity_preds = torch.argmax(identity_logits, dim=1)
                
                # Update metrics
                metrics.update(
                    gender_preds, identity_preds, gender_targets, identity_targets,
                    total_loss.item(), gender_loss.item(), identity_loss.item()
                )
        
        test_metrics = metrics.compute_metrics()
        
        print("\nTest Results:")
        print("-" * 50)
        print(f"Final Score: {test_metrics['final_score']:.4f}")
        print(f"Gender Classification:")
        print(f"  Accuracy: {test_metrics['gender_accuracy']:.4f}")
        print(f"  Precision: {test_metrics['gender_precision']:.4f}")
        print(f"  Recall: {test_metrics['gender_recall']:.4f}")
        print(f"  F1-Score: {test_metrics['gender_f1']:.4f}")
        print(f"Face Recognition:")
        print(f"  Accuracy: {test_metrics['identity_accuracy']:.4f}")
        print(f"  Precision: {test_metrics['identity_precision']:.4f}")
        print(f"  Recall: {test_metrics['identity_recall']:.4f}")
        print(f"  F1-Score: {test_metrics['identity_f1']:.4f}")
        
        return test_metrics
    
    def plot_training_history(self, save_path='training_history.png'):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plots
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Score plots
        axes[0, 1].plot(self.history['train_final_score'], label='Train Score')
        axes[0, 1].plot(self.history['val_final_score'], label='Val Score')
        axes[0, 1].set_title('Final Score Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training history saved: {save_path}")

def create_trainer(model, train_loader, val_loader, test_loader=None):
    """Factory function to create trainer"""
    return ICANTrainer(model, train_loader, val_loader, test_loader) 