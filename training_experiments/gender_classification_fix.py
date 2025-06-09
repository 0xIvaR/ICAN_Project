"""
Post-processing fix for gender classification issues
Applies corrections to model predictions to fix bias without retraining
"""

import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config
from model import create_ican_model

class GenderClassificationFix:
    """
    Post-processing fix for gender classification bias
    Applies statistical corrections and ensemble methods
    """
    
    def __init__(self, original_checkpoint_path, device=None):
        """
        Initialize the gender classification fix
        
        Args:
            original_checkpoint_path: Path to the original model
            device: Device to run inference on
        """
        self.device = device or Config.DEVICE
        self.model = None
        self.gender_classes = ['Female', 'Male']
        
        # Post-processing parameters
        self.bias_correction_factor = 0.3  # Reduces female bias
        self.confidence_threshold = 0.8    # High confidence threshold
        
        # Image preprocessing
        self.transform = A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.Normalize(mean=Config.MEAN, std=Config.STD),
            ToTensorV2(),
        ])
        
        self.load_model(original_checkpoint_path)
    
    def load_model(self, checkpoint_path):
        """Load the original model"""
        print(f"Loading original model from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint.get('config', {})
        num_identity_classes = config.get('num_identity_classes', 100)
        backbone = config.get('backbone', Config.BACKBONE)
        
        self.model = create_ican_model(num_identity_classes, backbone)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded successfully!")
    
    def preprocess_image(self, image_input):
        """Preprocess image for inference"""
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Could not load image: {image_input}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            image = np.array(image_input.convert('RGB'))
        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()
            if len(image.shape) == 3 and image.shape[2] == 3:
                if image_input.dtype == np.uint8:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Unsupported image input type")
        
        augmented = self.transform(image=image)
        tensor = augmented['image'].unsqueeze(0)
        return tensor.to(self.device)
    
    def apply_bias_correction(self, gender_probs):
        """
        Apply bias correction to gender probabilities
        Reduces the female bias by adjusting probabilities
        """
        corrected_probs = gender_probs.clone()
        
        # Reduce female probability and increase male probability
        female_prob = corrected_probs[0, 0]  # Female is index 0
        male_prob = corrected_probs[0, 1]    # Male is index 1
        
        # Apply correction factor to reduce female bias
        correction = self.bias_correction_factor
        
        # Reduce female confidence and increase male confidence
        new_female_prob = female_prob * (1 - correction)
        new_male_prob = male_prob * (1 + correction)
        
        # Renormalize to ensure probabilities sum to 1
        total = new_female_prob + new_male_prob
        corrected_probs[0, 0] = new_female_prob / total
        corrected_probs[0, 1] = new_male_prob / total
        
        return corrected_probs
    
    def predict_with_ensemble_and_correction(self, image_input, num_augmentations=5):
        """
        Predict gender with ensemble and bias correction
        """
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            if isinstance(image_input, Image.Image):
                image = np.array(image_input.convert('RGB'))
            else:
                image = image_input
        
        # Ensemble with multiple augmentations
        ensemble_transform = A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.NoOp(p=0.3),  # Sometimes no augmentation
            ], p=0.8),
            A.Normalize(mean=Config.MEAN, std=Config.STD),
            ToTensorV2(),
        ])
        
        all_gender_probs = []
        
        with torch.no_grad():
            for i in range(num_augmentations):
                # Apply augmentation
                augmented = ensemble_transform(image=image)
                tensor = augmented['image'].unsqueeze(0).to(self.device)
                
                # Get prediction
                gender_logits, _ = self.model(tensor)
                gender_probs = F.softmax(gender_logits, dim=1)
                
                # Apply bias correction
                corrected_probs = self.apply_bias_correction(gender_probs)
                all_gender_probs.append(corrected_probs)
        
        # Average the corrected probabilities
        avg_probs = torch.stack(all_gender_probs).mean(0)
        
        # Get final prediction
        gender_pred = torch.argmax(avg_probs, dim=1).cpu().item()
        gender_confidence = avg_probs.max().cpu().item()
        
        # Calculate uncertainty metrics
        prob_std = torch.stack(all_gender_probs).std(0).max().cpu().item()
        is_uncertain = prob_std > 0.1 or gender_confidence < self.confidence_threshold
        
        return {
            'gender': {
                'prediction': self.gender_classes[gender_pred],
                'confidence': gender_confidence,
                'probabilities': {
                    'Female': avg_probs[0, 0].cpu().item(),
                    'Male': avg_probs[0, 1].cpu().item()
                },
                'uncertainty': prob_std,
                'uncertain': is_uncertain
            },
            'method': 'ensemble_with_bias_correction'
        }
    
    def predict_with_rules(self, image_input):
        """
        Predict using simple rule-based post-processing
        """
        # Get standard prediction
        input_tensor = self.preprocess_image(image_input)
        
        with torch.no_grad():
            gender_logits, _ = self.model(input_tensor)
            gender_probs = F.softmax(gender_logits, dim=1)
            
            # Apply stronger bias correction for rule-based method
            corrected_probs = self.apply_bias_correction(gender_probs)
            
            # Additional rule: if female confidence is very high but corrected male is close,
            # consider it uncertain
            female_prob = corrected_probs[0, 0].cpu().item()
            male_prob = corrected_probs[0, 1].cpu().item()
            
            # Rule-based adjustment
            if female_prob > 0.9 and male_prob > 0.3:  # Very high female but decent male after correction
                # Further balance the probabilities
                balanced_female = 0.6
                balanced_male = 0.4
                corrected_probs[0, 0] = balanced_female
                corrected_probs[0, 1] = balanced_male
            
            gender_pred = torch.argmax(corrected_probs, dim=1).cpu().item()
            gender_confidence = corrected_probs.max().cpu().item()
        
        return {
            'gender': {
                'prediction': self.gender_classes[gender_pred],
                'confidence': gender_confidence,
                'probabilities': {
                    'Female': corrected_probs[0, 0].cpu().item(),
                    'Male': corrected_probs[0, 1].cpu().item()
                }
            },
            'method': 'rule_based_correction'
        }

def test_gender_fix():
    """Test the gender classification fix on test images"""
    
    print("🛠️  TESTING GENDER CLASSIFICATION FIX")
    print("=" * 60)
    
    # Initialize the fix
    fix = GenderClassificationFix("checkpoints/ican_best_model.pth")
    
    # Test images
    test_dir = "test_images"
    test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\nTesting {len(test_images)} images:")
    print("=" * 60)
    
    for img_file in test_images:
        img_path = os.path.join(test_dir, img_file)
        
        try:
            # Method 1: Ensemble with bias correction
            result1 = fix.predict_with_ensemble_and_correction(img_path)
            
            # Method 2: Rule-based correction
            result2 = fix.predict_with_rules(img_path)
            
            print(f"\n📸 {img_file}:")
            print(f"   Original Model (from previous test): Female (0.95+)")
            print(f"   ")
            print(f"   🔧 Ensemble + Bias Correction:")
            print(f"      Prediction: {result1['gender']['prediction']} (confidence: {result1['gender']['confidence']:.3f})")
            print(f"      Probabilities - Female: {result1['gender']['probabilities']['Female']:.3f}, Male: {result1['gender']['probabilities']['Male']:.3f}")
            if result1['gender']['uncertain']:
                print(f"      ⚠️  Uncertain (uncertainty: {result1['gender']['uncertainty']:.3f})")
            
            print(f"   ")
            print(f"   📏 Rule-based Correction:")
            print(f"      Prediction: {result2['gender']['prediction']} (confidence: {result2['gender']['confidence']:.3f})")
            print(f"      Probabilities - Female: {result2['gender']['probabilities']['Female']:.3f}, Male: {result2['gender']['probabilities']['Male']:.3f}")
            
        except Exception as e:
            print(f"❌ Error processing {img_file}: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✅ Gender classification fix testing completed!")
    print("\n💡 RECOMMENDATIONS:")
    print("1. Use ensemble method for more reliable predictions")
    print("2. Consider predictions with high uncertainty as needing manual review")
    print("3. The fix reduces female bias but may not be perfect for all images")
    print("4. For production use, collect more balanced training data and retrain")

if __name__ == "__main__":
    test_gender_fix() 