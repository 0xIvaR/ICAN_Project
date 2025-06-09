"""
Inference module for ICAN model
Provides functions for making predictions on new face images
"""

import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import Config
from src.model import create_ican_model

class ICANInference:
    """ICAN model inference class"""
    
    def __init__(self, checkpoint_path=None, device=None):
        """
        Initialize inference class
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.device = device or Config.DEVICE
        self.model = None
        self.gender_classes = ['Female', 'Male']  # Update based on your label encoding
        self.identity_classes = None
        
        # Image preprocessing
        self.transform = A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.Normalize(mean=Config.MEAN, std=Config.STD),
            ToTensorV2(),
        ])
        
        if checkpoint_path:
            self.load_model(checkpoint_path)
    
    def load_model(self, checkpoint_path):
        """Load trained model from checkpoint"""
        print(f"Loading model from: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get model configuration
        config = checkpoint.get('config', {})
        num_identity_classes = config.get('num_identity_classes', 100)
        backbone = config.get('backbone', Config.BACKBONE)
        
        # Create model
        self.model = create_ican_model(num_identity_classes, backbone)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Set identity classes (in practice, you'd load this from your dataset)
        self.identity_classes = [f'person_{i}' for i in range(num_identity_classes)]
        
        print(f"✓ Model loaded successfully!")
        print(f"  Backbone: {backbone}")
        print(f"  Identity classes: {num_identity_classes}")
        print(f"  Device: {self.device}")
    
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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            # PIL Image
            image = np.array(image_input.convert('RGB'))
        elif isinstance(image_input, np.ndarray):
            # Numpy array
            image = image_input
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume BGR format from OpenCV
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Unsupported image input type")
        
        # Apply preprocessing
        augmented = self.transform(image=image)
        tensor = augmented['image'].unsqueeze(0)  # Add batch dimension
        
        return tensor.to(self.device)
    
    def predict(self, image_input, return_probabilities=False):
        """
        Make prediction on a single image
        
        Args:
            image_input: Image input (path, PIL Image, or numpy array)
            return_probabilities: If True, return prediction probabilities
            
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
        
        result = {
            'gender': {
                'prediction': self.gender_classes[gender_pred],
                'confidence': gender_confidence
            },
            'identity': {
                'prediction': self.identity_classes[identity_pred],
                'confidence': identity_confidence
            }
        }
        
        if return_probabilities:
            result['gender']['probabilities'] = {
                self.gender_classes[i]: prob for i, prob in enumerate(gender_probs[0].cpu().numpy())
            }
            # Only return top-5 identity probabilities to avoid clutter
            top5_indices = torch.topk(identity_probs[0], 5).indices.cpu().numpy()
            result['identity']['top5_probabilities'] = {
                self.identity_classes[i]: identity_probs[0][i].cpu().item() for i in top5_indices
            }
        
        return result
    
    def predict_batch(self, image_list, batch_size=32):
        """
        Make predictions on a batch of images
        
        Args:
            image_list: List of image inputs
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = []
        
        for i in range(0, len(image_list), batch_size):
            batch_images = image_list[i:i + batch_size]
            batch_tensors = []
            
            # Preprocess batch
            for img in batch_images:
                tensor = self.preprocess_image(img)
                batch_tensors.append(tensor)
            
            # Stack into batch
            batch_tensor = torch.cat(batch_tensors, dim=0)
            
            # Make predictions
            with torch.no_grad():
                gender_logits, identity_logits = self.model(batch_tensor)
                
                gender_probs = F.softmax(gender_logits, dim=1)
                identity_probs = F.softmax(identity_logits, dim=1)
                
                gender_preds = torch.argmax(gender_probs, dim=1).cpu().numpy()
                identity_preds = torch.argmax(identity_probs, dim=1).cpu().numpy()
                
                gender_confidences = gender_probs.max(dim=1)[0].cpu().numpy()
                identity_confidences = identity_probs.max(dim=1)[0].cpu().numpy()
            
            # Format results
            for j in range(len(batch_images)):
                result = {
                    'gender': {
                        'prediction': self.gender_classes[gender_preds[j]],
                        'confidence': float(gender_confidences[j])
                    },
                    'identity': {
                        'prediction': self.identity_classes[identity_preds[j]],
                        'confidence': float(identity_confidences[j])
                    }
                }
                results.append(result)
        
        return results
    
    def get_embeddings(self, image_input):
        """
        Get feature embeddings for an image
        
        Args:
            image_input: Image input (path, PIL Image, or numpy array)
            
        Returns:
            Feature embeddings as numpy array
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess image
        input_tensor = self.preprocess_image(image_input)
        
        # Get embeddings
        embeddings = self.model.get_embedding(input_tensor)
        
        return embeddings.cpu().numpy()
    
    def compare_faces(self, image1, image2, threshold=0.5):
        """
        Compare two faces and determine if they are the same person
        
        Args:
            image1: First image
            image2: Second image
            threshold: Similarity threshold (cosine similarity)
            
        Returns:
            Dictionary with comparison result
        """
        # Get embeddings for both images
        emb1 = self.get_embeddings(image1)
        emb2 = self.get_embeddings(image2)
        
        # Calculate cosine similarity
        similarity = np.dot(emb1[0], emb2[0]) / (np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0]))
        
        # Determine if same person
        is_same_person = similarity > threshold
        
        return {
            'similarity': float(similarity),
            'is_same_person': is_same_person,
            'threshold': threshold
        }

def load_inference_model(checkpoint_path=None):
    """
    Convenience function to load inference model
    
    Args:
        checkpoint_path: Path to model checkpoint
        
    Returns:
        ICANInference instance
    """
    if checkpoint_path is None:
        checkpoint_path = Config.BEST_MODEL_PATH
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return ICANInference(checkpoint_path)

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ICAN Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--compare', type=str, default=None, help='Path to second image for comparison')
    
    args = parser.parse_args()
    
    # Load model
    inference = load_inference_model(args.checkpoint)
    
    # Make prediction
    result = inference.predict(args.image, return_probabilities=True)
    
    print("Prediction Results:")
    print(f"Gender: {result['gender']['prediction']} (confidence: {result['gender']['confidence']:.3f})")
    print(f"Identity: {result['identity']['prediction']} (confidence: {result['identity']['confidence']:.3f})")
    
    if 'probabilities' in result['gender']:
        print("\nGender Probabilities:")
        for gender, prob in result['gender']['probabilities'].items():
            print(f"  {gender}: {prob:.3f}")
    
    if 'top5_probabilities' in result['identity']:
        print("\nTop-5 Identity Probabilities:")
        for identity, prob in result['identity']['top5_probabilities'].items():
            print(f"  {identity}: {prob:.3f}")
    
    # Face comparison if second image provided
    if args.compare:
        comparison = inference.compare_faces(args.image, args.compare)
        print(f"\nFace Comparison:")
        print(f"Similarity: {comparison['similarity']:.3f}")
        print(f"Same Person: {comparison['is_same_person']}")
        print(f"Threshold: {comparison['threshold']}") 