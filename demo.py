"""
ICAN Demo Script
Demonstrates the capabilities of the ICAN model for face recognition and gender classification
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

from src.config import Config
from src.model import create_ican_model, count_parameters
from inference import ICANInference
from src.dataset import create_data_splits

def demo_model_architecture():
    """Demonstrate model architecture and capabilities"""
    print("🏗️  ICAN Model Architecture Demo")
    print("=" * 60)
    
    # Create model
    print("Creating ICAN model...")
    model = create_ican_model(num_identity_classes=100)
    
    # Model info
    total_params = count_parameters(model)
    print(f"✓ Model: {Config.MODEL_NAME}")
    print(f"✓ Backbone: {Config.BACKBONE}")
    print(f"✓ Input Size: {Config.IMG_SIZE}x{Config.IMG_SIZE}")
    print(f"✓ Total Parameters: {total_params:,}")
    print(f"✓ Device: {Config.DEVICE}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(4, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    with torch.no_grad():
        gender_out, identity_out = model(dummy_input)
    
    print(f"✓ Input shape: {dummy_input.shape}")
    print(f"✓ Gender output shape: {gender_out.shape}")
    print(f"✓ Identity output shape: {identity_out.shape}")
    
    # Test embeddings
    print("\nTesting feature embeddings...")
    with torch.no_grad():
        embeddings = model.get_embedding(dummy_input)
    print(f"✓ Embedding shape: {embeddings.shape}")
    print(f"✓ Embedding norm: {torch.norm(embeddings, dim=1).mean():.4f}")
    
    print("\n✅ Model architecture demo completed!")
    return model

def demo_robust_augmentations():
    """Demonstrate robust augmentation pipeline"""
    print("\n🌟 Robust Augmentation Demo")
    print("=" * 60)
    
    from src.dataset import get_robust_transforms
    
    # Create sample image
    sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Get transforms
    train_transform = get_robust_transforms('train', Config.IMG_SIZE)
    val_transform = get_robust_transforms('val', Config.IMG_SIZE)
    
    print("✓ Training augmentations:")
    for transform in train_transform.transforms:
        print(f"  - {transform.__class__.__name__}")
    
    print("✓ Validation transforms:")
    for transform in val_transform.transforms:
        print(f"  - {transform.__class__.__name__}")
    
    # Apply augmentations
    print("\nTesting augmentation pipeline...")
    augmented = train_transform(image=sample_image)
    print(f"✓ Input shape: {sample_image.shape}")
    print(f"✓ Output shape: {augmented['image'].shape}")
    
    print("\n✅ Augmentation demo completed!")

def demo_sample_data():
    """Demonstrate sample data creation and loading"""
    print("\n📊 Sample Data Demo")
    print("=" * 60)
    
    # Create sample data
    print("Creating sample FACECOM dataset...")
    train_df, val_df, test_df = create_data_splits("data/sample_images")
    
    print(f"✓ Training samples: {len(train_df)}")
    print(f"✓ Validation samples: {len(val_df)}")
    print(f"✓ Test samples: {len(test_df)}")
    print(f"✓ Unique identities: {len(train_df.person_id.unique())}")
    print(f"✓ Gender distribution:")
    print(train_df.gender.value_counts().to_string())
    
    print("\n✅ Sample data demo completed!")
    return len(train_df.person_id.unique())

def demo_inference_capabilities():
    """Demonstrate inference capabilities"""
    print("\n🔍 Inference Capabilities Demo")
    print("=" * 60)
    
    # Note: This demo shows the inference interface
    # In practice, you would load a trained model
    
    print("Inference capabilities include:")
    print("✓ Single image prediction")
    print("✓ Batch prediction")
    print("✓ Feature embedding extraction")
    print("✓ Face comparison/verification")
    print("✓ Probability distributions")
    
    # Example inference code (would work with trained model)
    print("\nExample inference usage:")
    print("""
from inference import load_inference_model

# Load trained model
model = load_inference_model("checkpoints/ican_best_model.pth")

# Make prediction
result = model.predict("path/to/image.jpg", return_probabilities=True)
print(f"Gender: {result['gender']['prediction']}")
print(f"Identity: {result['identity']['prediction']}")

# Compare faces
comparison = model.compare_faces("image1.jpg", "image2.jpg")
print(f"Same person: {comparison['is_same_person']}")
    """)
    
    print("\n✅ Inference demo completed!")

def demo_training_configuration():
    """Demonstrate training configuration options"""
    print("\n⚙️  Training Configuration Demo")
    print("=" * 60)
    
    print(f"Model Configuration:")
    print(f"  Model Name: {Config.MODEL_NAME}")
    print(f"  Backbone: {Config.BACKBONE}")
    print(f"  Image Size: {Config.IMG_SIZE}")
    print(f"  Pretrained: {Config.PRETRAINED}")
    
    print(f"\nTraining Configuration:")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Learning Rate: {Config.LEARNING_RATE}")
    print(f"  Weight Decay: {Config.WEIGHT_DECAY}")
    print(f"  Max Epochs: {Config.NUM_EPOCHS}")
    print(f"  Early Stopping Patience: {Config.PATIENCE}")
    
    print(f"\nMulti-task Configuration:")
    print(f"  Gender Task Weight: {Config.GENDER_WEIGHT}")
    print(f"  Identity Task Weight: {Config.IDENTITY_WEIGHT}")
    print(f"  Final Score = {Config.GENDER_WEIGHT} × Gender + {Config.IDENTITY_WEIGHT} × Identity")
    
    print(f"\nAugmentation Configuration:")
    print(f"  Blur Probability: {Config.BLUR_PROB}")
    print(f"  Noise Probability: {Config.NOISE_PROB}")
    print(f"  Brightness Probability: {Config.BRIGHTNESS_PROB}")
    
    print("\n✅ Configuration demo completed!")

def demo_robustness_features():
    """Demonstrate robustness features"""
    print("\n🛡️  Robustness Features Demo")
    print("=" * 60)
    
    print("ICAN Robustness Features:")
    print("\n1. Adverse Condition Handling:")
    print("   ✓ Motion blur resistance")
    print("   ✓ Low-light performance")
    print("   ✓ Weather condition adaptation (fog, rain)")
    print("   ✓ Overexposure/glare handling")
    print("   ✓ Noise robustness")
    
    print("\n2. Architectural Features:")
    print("   ✓ Spatial attention mechanisms")
    print("   ✓ Condition-adaptive blocks")
    print("   ✓ Multi-scale feature extraction")
    print("   ✓ Residual connections")
    print("   ✓ Batch normalization")
    
    print("\n3. Training Techniques:")
    print("   ✓ Multi-task learning")
    print("   ✓ Data augmentation")
    print("   ✓ Label smoothing")
    print("   ✓ Gradient clipping")
    print("   ✓ Early stopping")
    
    print("\n4. Evaluation Metrics:")
    print("   ✓ Cross-validation")
    print("   ✓ Condition-specific evaluation")
    print("   ✓ Top-k accuracy")
    print("   ✓ Confusion matrices")
    print("   ✓ ROC curves")
    
    print("\n✅ Robustness features demo completed!")

def main():
    """Run complete ICAN demo"""
    print("🌟" * 30)
    print("ICAN - Intelligent Condition-Adaptive Network")
    print("Robust Face Recognition and Gender Classification Demo")
    print("🌟" * 30)
    
    try:
        # 1. Model Architecture Demo
        model = demo_model_architecture()
        
        # 2. Augmentation Demo
        demo_robust_augmentations()
        
        # 3. Sample Data Demo
        num_identities = demo_sample_data()
        
        # 4. Training Configuration Demo
        demo_training_configuration()
        
        # 5. Robustness Features Demo
        demo_robustness_features()
        
        # 6. Inference Demo
        demo_inference_capabilities()
        
        # Final Summary
        print("\n🎉 ICAN Demo Summary")
        print("=" * 60)
        print("✅ All demos completed successfully!")
        print(f"🏗️  Model: {Config.MODEL_NAME} with {count_parameters(model):,} parameters")
        print(f"📊 Sample dataset: {num_identities} identities")
        print(f"🛡️  Robust to: blur, fog, rain, low-light, overexposure")
        print(f"🎯 Tasks: Gender classification + Face recognition")
        print(f"⚡ Ready for training and inference!")
        
        print("\n🚀 Next Steps:")
        print("1. Prepare your FACECOM dataset")
        print("2. Run: python main.py --epochs 10 --batch_size 16 (for quick test)")
        print("3. For full training: python main.py")
        print("4. For inference: python inference.py --image your_image.jpg")
        
        print("\n📚 For more information, see README.md")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "🌟" * 30)

if __name__ == "__main__":
    main() 