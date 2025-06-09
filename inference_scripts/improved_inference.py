"""
Improved inference script for the retrained ICAN model
Uses the cleaned labels and improved model for better gender classification
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

class ImprovedICANInference:
    """Improved ICAN model inference class with better gender classification"""
    
    def __init__(self, checkpoint_path=None, device=None):
        """
        Initialize inference class
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.device = device or Config.DEVICE
        self.model = None
        self.gender_classes = ['Female', 'Male']  # 0: Female, 1: Male
        self.identity_classes = None
        self.gender_mapping = None  # For consistent labeling
        
        # Image preprocessing - same as training
        self.transform = A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.Normalize(mean=Config.MEAN, std=Config.STD),
            ToTensorV2(),
        ])
        
        if checkpoint_path:
            self.load_model(checkpoint_path)
    
    def load_model(self, checkpoint_path):
        """Load improved trained model from checkpoint"""
        print(f"Loading improved model from: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get model configuration
        config = checkpoint.get('config', {})
        num_identity_classes = config.get('num_identity_classes', 100)
        backbone = config.get('backbone', Config.BACKBONE)
        self.gender_mapping = config.get('gender_mapping', {})
        
        # Create model
        self.model = create_ican_model(num_identity_classes, backbone)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Set identity classes
        self.identity_classes = [f'person_{i}' for i in range(num_identity_classes)]
        
        print(f"✓ Improved model loaded successfully!")
        print(f"  Backbone: {backbone}")
        print(f"  Identity classes: {num_identity_classes}")
        best_acc = checkpoint.get('best_val_accuracy', None)
        if best_acc is not None:
            print(f"  Best validation accuracy: {best_acc:.2f}%")
        else:
            print(f"  Best validation accuracy: N/A")
        print(f"  Device: {self.device}")
        print(f"  Gender mapping applied: {len(self.gender_mapping)} people")
    
    def preprocess_image(self, image_input):
        """
        Preprocess image for inference
        
        Args:
            image_input: Can be path to image file, PIL Image, or numpy array
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Load image
        if isinstance(image_input, str):
            # Image path
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Could not load image: {image_input}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            # PIL Image
            image = np.array(image_input.convert('RGB'))
        elif isinstance(image_input, np.ndarray):
            # Numpy array
            image = image_input.copy()
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Check if BGR format from OpenCV
                if image_input.dtype == np.uint8:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Unsupported image input type")
        
        # Apply preprocessing
        augmented = self.transform(image=image)
        tensor = augmented['image'].unsqueeze(0)  # Add batch dimension
        
        return tensor.to(self.device)
    
    def predict(self, image_input, return_probabilities=False, confidence_threshold=0.6):
        """
        Make prediction on a single image with improved confidence handling
        
        Args:
            image_input: Image input (path, PIL Image, or numpy array)
            return_probabilities: If True, return prediction probabilities
            confidence_threshold: Minimum confidence for reliable prediction
            
        Returns:
            Dictionary with predictions and optionally probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess image
        input_tensor = self.preprocess_image(image_input)
        
        # Make prediction
        with torch.no_grad():
            gender_logits, identity_logits = self.model(input_tensor)
            
            # Get probabilities
            gender_probs = F.softmax(gender_logits, dim=1)
            identity_probs = F.softmax(identity_logits, dim=1)
            
            # Get predictions
            gender_pred = torch.argmax(gender_probs, dim=1).cpu().item()
            identity_pred = torch.argmax(identity_probs, dim=1).cpu().item()
            
            # Get confidence scores
            gender_confidence = gender_probs.max().cpu().item()
            identity_confidence = identity_probs.max().cpu().item()
        
        # Determine prediction reliability
        gender_reliable = gender_confidence >= confidence_threshold
        
        result = {
            'gender': {
                'prediction': self.gender_classes[gender_pred],
                'confidence': gender_confidence,
                'reliable': gender_reliable
            },
            'identity': {
                'prediction': self.identity_classes[identity_pred],
                'confidence': identity_confidence
            }
        }
        
        # Add warning for low confidence predictions
        if not gender_reliable:
            result['gender']['warning'] = f"Low confidence ({gender_confidence:.3f} < {confidence_threshold})"
        
        if return_probabilities:
            result['gender']['probabilities'] = {
                self.gender_classes[i]: prob for i, prob in enumerate(gender_probs[0].cpu().numpy())
            }
            # Only return top-5 identity probabilities
            top5_indices = torch.topk(identity_probs[0], min(5, len(self.identity_classes))).indices.cpu().numpy()
            result['identity']['top5_probabilities'] = {
                self.identity_classes[i]: identity_probs[0][i].cpu().item() for i in top5_indices
            }
        
        return result
    
    def test_with_ensemble(self, image_input, num_augmentations=5):
        """
        Test with multiple augmentations for more robust prediction
        
        Args:
            image_input: Image input
            num_augmentations: Number of augmented versions to test
            
        Returns:
            Ensemble prediction results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Load original image
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Convert to numpy array
            if isinstance(image_input, Image.Image):
                image = np.array(image_input.convert('RGB'))
            else:
                image = image_input
        
        # Create augmentation pipeline for ensemble
        ensemble_transform = A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            ], p=0.7),
            A.Normalize(mean=Config.MEAN, std=Config.STD),
            ToTensorV2(),
        ])
        
        gender_predictions = []
        gender_confidences = []
        
        with torch.no_grad():
            for i in range(num_augmentations):
                # Apply augmentation
                augmented = ensemble_transform(image=image)
                tensor = augmented['image'].unsqueeze(0).to(self.device)
                
                # Make prediction
                gender_logits, _ = self.model(tensor)
                gender_probs = F.softmax(gender_logits, dim=1)
                
                gender_pred = torch.argmax(gender_probs, dim=1).cpu().item()
                gender_conf = gender_probs.max().cpu().item()
                
                gender_predictions.append(gender_pred)
                gender_confidences.append(gender_conf)
        
        # Ensemble results
        most_common_pred = max(set(gender_predictions), key=gender_predictions.count)
        avg_confidence = np.mean(gender_confidences)
        prediction_consistency = gender_predictions.count(most_common_pred) / len(gender_predictions)
        
        return {
            'ensemble_prediction': self.gender_classes[most_common_pred],
            'average_confidence': avg_confidence,
            'prediction_consistency': prediction_consistency,
            'individual_predictions': [self.gender_classes[p] for p in gender_predictions],
            'individual_confidences': gender_confidences
        }

def test_improved_model():
    """Test the improved model on test images"""
    
    print("🧪 TESTING IMPROVED ICAN MODEL")
    print("=" * 60)
    
    # Initialize improved inference
    improved_checkpoint = "checkpoints/ican_improved_model.pth" 
    original_checkpoint = "checkpoints/ican_best_model.pth"
    
    # Check which checkpoint exists
    if os.path.exists(improved_checkpoint):
        print("Loading improved model...")
        inference = ImprovedICANInference(improved_checkpoint)
    else:
        print("Improved model not found, using original model...")
        inference = ImprovedICANInference(original_checkpoint)
    
    # Test images
    test_dir = "test_images"
    test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\nTesting {len(test_images)} images:")
    print("=" * 60)
    
    for img_file in test_images:
        img_path = os.path.join(test_dir, img_file)
        
        try:
            # Standard prediction
            result = inference.predict(img_path, return_probabilities=True)
            
            # Ensemble prediction for better reliability
            ensemble = inference.test_with_ensemble(img_path, num_augmentations=5)
            
            print(f"\n📸 {img_file}:")
            print(f"   Standard Prediction: {result['gender']['prediction']} (confidence: {result['gender']['confidence']:.3f})")
            if 'warning' in result['gender']:
                print(f"   ⚠️  {result['gender']['warning']}")
            
            print(f"   Ensemble Prediction: {ensemble['ensemble_prediction']} (avg confidence: {ensemble['average_confidence']:.3f})")
            print(f"   Prediction Consistency: {ensemble['prediction_consistency']:.1%}")
            print(f"   Individual Predictions: {ensemble['individual_predictions']}")
            
            # Gender probabilities
            print(f"   Gender Probabilities:")
            for gender, prob in result['gender']['probabilities'].items():
                print(f"     {gender}: {prob:.3f}")
            
        except Exception as e:
            print(f"❌ Error processing {img_file}: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✅ Testing completed!")

if __name__ == "__main__":
    test_improved_model() 