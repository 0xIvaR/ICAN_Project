"""
Main training script for ICAN - Intelligent Condition-Adaptive Network
Robust Face Recognition and Gender Classification under Adverse Visual Conditions
"""

import os
import argparse
import torch
import numpy as np
import random
from datetime import datetime

from src.config import Config
from src.dataset import create_data_splits, get_data_loaders
from src.model import create_ican_model, count_parameters
from src.trainer import create_trainer

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_directories():
    """Create necessary directories"""
    os.makedirs("data", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

def print_system_info():
    """Print system information"""
    print("=" * 80)
    print("ICAN - Intelligent Condition-Adaptive Network")
    print("Robust Face Recognition and Gender Classification")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {Config.DEVICE}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Backbone: {Config.BACKBONE}")
    print(f"Image Size: {Config.IMG_SIZE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")
    print(f"Max Epochs: {Config.NUM_EPOCHS}")
    print("=" * 80)

def create_sample_data():
    """Create sample FACECOM data for demonstration"""
    print("Creating sample FACECOM dataset...")
    
    # Create sample data splits
    train_df, val_df, test_df = create_data_splits("data/sample_images")
    
    print(f"✓ Sample dataset created:")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")
    print(f"  Test samples: {len(test_df)}")
    
    return len(train_df.person_id.unique())

def main():
    parser = argparse.ArgumentParser(description='Train ICAN model')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--backbone', type=str, default=None, help='Backbone model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--evaluate_only', action='store_true', help='Only evaluate the model')
    parser.add_argument('--data_path', type=str, default=None, help='Path to FACECOM dataset')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup directories
    setup_directories()
    
    # Print system information
    print_system_info()
    
    # Override config with command line arguments
    if args.epochs is not None:
        Config.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        Config.LEARNING_RATE = args.lr
    if args.backbone is not None:
        Config.BACKBONE = args.backbone
    if args.data_path is not None:
        Config.DATA_ROOT = args.data_path
    
    try:
        # Check if actual FACECOM dataset exists
        if not os.path.exists(Config.DATA_ROOT) or not os.path.exists(Config.TRAIN_CSV):
            print("⚠️  FACECOM dataset not found. Creating sample data for demonstration.")
            print("   To use the actual dataset, place FACECOM data in 'data/FACECOM/'")
            print("   and ensure train_labels.csv, val_labels.csv, test_labels.csv exist.")
            num_identity_classes = create_sample_data()
        else:
            print("✓ FACECOM dataset found!")
            # Get number of identity classes from training data
            import pandas as pd
            train_df = pd.read_csv(Config.TRAIN_CSV)
            num_identity_classes = len(train_df.person_id.unique())
            print(f"  Number of unique identities: {num_identity_classes}")
        
        # Get data loaders
        print("\n📊 Loading data...")
        train_loader, val_loader, test_loader, actual_num_classes = get_data_loaders(Config.BATCH_SIZE)
        
        if actual_num_classes is not None:
            num_identity_classes = actual_num_classes
        
        print(f"✓ Data loaders created:")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        print(f"  Identity classes: {num_identity_classes}")
        
        # Create model
        print(f"\n🏗️  Creating {Config.MODEL_NAME} model...")
        model = create_ican_model(num_identity_classes, Config.BACKBONE)
        total_params = count_parameters(model)
        print(f"✓ Model created with {total_params:,} trainable parameters")
        
        # Create trainer
        print("\n🎯 Setting up trainer...")
        trainer = create_trainer(model, train_loader, val_loader, test_loader)
        print("✓ Trainer initialized")
        
        # Resume from checkpoint if specified
        if args.resume:
            print(f"\n📂 Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        if args.evaluate_only:
            # Only evaluate the model
            if args.resume or os.path.exists(Config.BEST_MODEL_PATH):
                checkpoint_path = args.resume or Config.BEST_MODEL_PATH
                print(f"\n📊 Loading model for evaluation: {checkpoint_path}")
                trainer.load_checkpoint(checkpoint_path)
                test_metrics = trainer.evaluate()
                return test_metrics
            else:
                print("❌ No checkpoint found for evaluation!")
                return None
        
        # Start training
        print(f"\n🚀 Starting training for {Config.NUM_EPOCHS} epochs...")
        print(f"💾 Checkpoints will be saved to: {Config.CHECKPOINT_DIR}")
        print(f"🏆 Best model will be saved to: {Config.BEST_MODEL_PATH}")
        
        # Train the model
        history = trainer.train()
        
        print("\n🎉 Training completed!")
        
        # Plot training history
        print("\n📈 Generating training plots...")
        trainer.plot_training_history('results/training_history.png')
        
        # Evaluate on test set
        print("\n🧪 Evaluating on test set...")
        test_metrics = trainer.evaluate()
        
        # Save final results
        import json
        results = {
            'config': {
                'model_name': Config.MODEL_NAME,
                'backbone': Config.BACKBONE,
                'img_size': Config.IMG_SIZE,
                'batch_size': Config.BATCH_SIZE,
                'learning_rate': Config.LEARNING_RATE,
                'epochs_trained': len(history['train_loss']),
                'total_parameters': total_params,
                'num_identity_classes': num_identity_classes
            },
            'test_metrics': test_metrics,
            'training_history': dict(history)
        }
        
        with open('results/final_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: results/final_results.json")
        print(f"📈 Training plots saved to: results/training_history.png")
        print(f"🏆 Best model saved to: {Config.BEST_MODEL_PATH}")
        
        # Print final summary
        print("\n" + "=" * 80)
        print("FINAL RESULTS SUMMARY")
        print("=" * 80)
        if test_metrics:
            print(f"🎯 Final Score: {test_metrics['final_score']:.4f}")
            print(f"👤 Gender Classification Accuracy: {test_metrics['gender_accuracy']:.4f}")
            print(f"🔍 Face Recognition Accuracy: {test_metrics['identity_accuracy']:.4f}")
            print(f"📊 Gender F1-Score: {test_metrics['gender_f1']:.4f}")
            print(f"📊 Identity F1-Score: {test_metrics['identity_f1']:.4f}")
        print("=" * 80)
        
        return test_metrics
        
    except KeyboardInterrupt:
        print("\n❌ Training interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Example usage:
    # python main.py --epochs 50 --batch_size 16 --lr 0.001
    # python main.py --resume checkpoints/ican_best_model.pth --evaluate_only
    main() 