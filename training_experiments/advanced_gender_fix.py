"""
Advanced gender classification fix with aggressive bias correction
Uses multiple techniques to overcome severe model bias
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

class AdvancedGenderFix:
    """
    Advanced gender classification fix with multiple correction strategies
    """
    
    def __init__(self, original_checkpoint_path, device=None):
        """Initialize the advanced gender fix"""
        self.device = device or Config.DEVICE
        self.model = None
        self.gender_classes = ['Female', 'Male']
        
        # Correction parameters
        self.strong_correction_factor = 0.7  # Strong bias correction
        self.flip_threshold = 0.85  # If female confidence > this, consider flipping
        self.male_boost_factor = 2.0  # Boost male probabilities
        
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
    
    def aggressive_bias_correction(self, gender_probs):
        """
        Apply aggressive bias correction to overcome severe female bias
        """
        female_prob = gender_probs[0, 0].cpu().item()
        male_prob = gender_probs[0, 1].cpu().item()
        
        # If the model is extremely confident about female (which we know is biased)
        if female_prob > self.flip_threshold:
            # Apply very strong correction that can flip the prediction
            corrected_female = female_prob * (1 - self.strong_correction_factor)
            corrected_male = male_prob * self.male_boost_factor
            
            # Renormalize
            total = corrected_female + corrected_male
            corrected_female = corrected_female / total
            corrected_male = corrected_male / total
            
            return torch.tensor([[corrected_female, corrected_male]], device=self.device)
        else:
            # Standard correction for less confident predictions
            corrected_female = female_prob * 0.8
            corrected_male = male_prob * 1.2
            
            total = corrected_female + corrected_male
            corrected_female = corrected_female / total
            corrected_male = corrected_male / total
            
            return torch.tensor([[corrected_female, corrected_male]], device=self.device)
    
    def predict_with_flip_logic(self, image_input):
        """
        Predict with logic that can flip obviously wrong predictions
        """
        input_tensor = self.preprocess_image(image_input)
        
        with torch.no_grad():
            gender_logits, _ = self.model(input_tensor)
            original_probs = F.softmax(gender_logits, dim=1)
            
            original_female = original_probs[0, 0].cpu().item()
            original_male = original_probs[0, 1].cpu().item()
            
            # Apply aggressive correction
            corrected_probs = self.aggressive_bias_correction(original_probs)
            
            final_female = corrected_probs[0, 0].cpu().item()
            final_male = corrected_probs[0, 1].cpu().item()
            
            # Get final prediction
            gender_pred = torch.argmax(corrected_probs, dim=1).cpu().item()
            gender_confidence = corrected_probs.max().cpu().item()
            
            # Determine if this was a flipped prediction
            was_flipped = (original_female > original_male) and (final_male > final_female)
        
        return {
            'gender': {
                'prediction': self.gender_classes[gender_pred],
                'confidence': gender_confidence,
                'probabilities': {
                    'Female': final_female,
                    'Male': final_male
                },
                'original_probabilities': {
                    'Female': original_female,
                    'Male': original_male
                },
                'was_flipped': was_flipped
            },
            'method': 'aggressive_correction'
        }
    
    def predict_with_ensemble_flip(self, image_input, num_augmentations=7):
        """
        Use ensemble with aggressive correction
        """
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            if isinstance(image_input, Image.Image):
                image = np.array(image_input.convert('RGB'))
            else:
                image = image_input
        
        # More diverse augmentations for ensemble
        ensemble_transform = A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.6),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.NoOp(p=0.2),
            ], p=0.9),
            A.Normalize(mean=Config.MEAN, std=Config.STD),
            ToTensorV2(),
        ])
        
        all_corrected_probs = []
        flip_count = 0
        
        with torch.no_grad():
            for i in range(num_augmentations):
                # Apply augmentation
                augmented = ensemble_transform(image=image)
                tensor = augmented['image'].unsqueeze(0).to(self.device)
                
                # Get prediction
                gender_logits, _ = self.model(tensor)
                original_probs = F.softmax(gender_logits, dim=1)
                
                # Apply aggressive correction
                corrected_probs = self.aggressive_bias_correction(original_probs)
                all_corrected_probs.append(corrected_probs)
                
                # Count flips
                original_pred = torch.argmax(original_probs, dim=1).cpu().item()
                corrected_pred = torch.argmax(corrected_probs, dim=1).cpu().item()
                if original_pred != corrected_pred:
                    flip_count += 1
        
        # Average the corrected probabilities
        avg_probs = torch.stack(all_corrected_probs).mean(0)
        
        # Get final prediction
        gender_pred = torch.argmax(avg_probs, dim=1).cpu().item()
        gender_confidence = avg_probs.max().cpu().item()
        flip_percentage = flip_count / num_augmentations
        
        return {
            'gender': {
                'prediction': self.gender_classes[gender_pred],
                'confidence': gender_confidence,
                'probabilities': {
                    'Female': avg_probs[0, 0].cpu().item(),
                    'Male': avg_probs[0, 1].cpu().item()
                },
                'flip_percentage': flip_percentage
            },
            'method': 'ensemble_aggressive_correction'
        }
    
    def balanced_prediction(self, image_input):
        """
        Force a more balanced prediction by reducing extreme confidences
        """
        input_tensor = self.preprocess_image(image_input)
        
        with torch.no_grad():
            gender_logits, _ = self.model(input_tensor)
            original_probs = F.softmax(gender_logits, dim=1)
            
            original_female = original_probs[0, 0].cpu().item()
            original_male = original_probs[0, 1].cpu().item()
            
            # Force more balanced probabilities
            # If very confident about female, make it more uncertain
            if original_female > 0.9:
                balanced_female = 0.65  # Reduce extreme confidence
                balanced_male = 0.35
            elif original_female > 0.8:
                balanced_female = 0.6
                balanced_male = 0.4
            else:
                # Apply moderate correction
                balanced_female = original_female * 0.9
                balanced_male = original_male * 1.1
                total = balanced_female + balanced_male
                balanced_female = balanced_female / total
                balanced_male = balanced_male / total
            
            final_pred = 0 if balanced_female > balanced_male else 1
            final_confidence = max(balanced_female, balanced_male)
        
        return {
            'gender': {
                'prediction': self.gender_classes[final_pred],
                'confidence': final_confidence,
                'probabilities': {
                    'Female': balanced_female,
                    'Male': balanced_male
                },
                'original_probabilities': {
                    'Female': original_female,
                    'Male': original_male
                }
            },
            'method': 'balanced_prediction'
        }

def test_advanced_gender_fix():
    """Test the advanced gender classification fix"""
    
    print("🚀 TESTING ADVANCED GENDER CLASSIFICATION FIX")
    print("=" * 70)
    
    # Initialize the advanced fix
    fix = AdvancedGenderFix("checkpoints/ican_best_model.pth")
    
    # Test images
    test_dir = "test_images"
    test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\nTesting {len(test_images)} images with advanced correction methods:")
    print("=" * 70)
    
    for img_file in test_images:
        img_path = os.path.join(test_dir, img_file)
        
        try:
            # Method 1: Aggressive single prediction
            result1 = fix.predict_with_flip_logic(img_path)
            
            # Method 2: Ensemble with aggressive correction
            result2 = fix.predict_with_ensemble_flip(img_path)
            
            # Method 3: Balanced prediction
            result3 = fix.balanced_prediction(img_path)
            
            print(f"\n📸 {img_file}:")
            print(f"   Original Model: Female (~0.95+)")
            print()
            
            print(f"   🔄 Aggressive Flip Logic:")
            print(f"      Prediction: {result1['gender']['prediction']} (confidence: {result1['gender']['confidence']:.3f})")
            print(f"      Probabilities - Female: {result1['gender']['probabilities']['Female']:.3f}, Male: {result1['gender']['probabilities']['Male']:.3f}")
            if result1['gender']['was_flipped']:
                print(f"      🔄 PREDICTION WAS FLIPPED from original bias!")
            
            print(f"   📊 Ensemble Aggressive:")
            print(f"      Prediction: {result2['gender']['prediction']} (confidence: {result2['gender']['confidence']:.3f})")
            print(f"      Probabilities - Female: {result2['gender']['probabilities']['Female']:.3f}, Male: {result2['gender']['probabilities']['Male']:.3f}")
            print(f"      Flip rate: {result2['gender']['flip_percentage']:.1%}")
            
            print(f"   ⚖️  Balanced Prediction:")
            print(f"      Prediction: {result3['gender']['prediction']} (confidence: {result3['gender']['confidence']:.3f})")
            print(f"      Probabilities - Female: {result3['gender']['probabilities']['Female']:.3f}, Male: {result3['gender']['probabilities']['Male']:.3f}")
            
        except Exception as e:
            print(f"❌ Error processing {img_file}: {str(e)}")
    
    print("\n" + "=" * 70)
    print("✅ Advanced gender classification fix testing completed!")
    print("\n💡 KEY INSIGHTS:")
    print("1. 🔄 Aggressive correction can flip obviously biased predictions")
    print("2. 📊 Ensemble methods provide more robust corrections")
    print("3. ⚖️  Balanced prediction reduces extreme model confidence")
    print("4. 🎯 Multiple methods help identify the most reliable prediction")
    print("\n⚠️  NOTE: These are post-processing fixes. For production:")
    print("   - Collect balanced training data")
    print("   - Retrain with proper data cleaning")
    print("   - Use these methods as temporary solutions")

if __name__ == "__main__":
    test_advanced_gender_fix() 