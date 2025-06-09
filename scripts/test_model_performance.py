"""
Test script to evaluate ICAN model performance on test images
"""

import os
import torch
from inference import ICANInference
import matplotlib.pyplot as plt
from PIL import Image

def test_model_with_images():
    """Test the model with all available test images"""
    
    # Initialize inference
    print("Loading ICAN model...")
    inference = ICANInference("checkpoints/ican_best_model.pth")
    
    # Test image directory
    test_dir = "test_images"
    test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\nTesting with {len(test_images)} images:")
    print("=" * 60)
    
    results = []
    
    for img_file in test_images:
        img_path = os.path.join(test_dir, img_file)
        
        try:
            # Make prediction
            result = inference.predict(img_path, return_probabilities=True)
            
            # Store result
            results.append({
                'filename': img_file,
                'gender_pred': result['gender']['prediction'],
                'gender_conf': result['gender']['confidence'],
                'gender_probs': result['gender']['probabilities'],
                'identity_pred': result['identity']['prediction'],
                'identity_conf': result['identity']['confidence']
            })
            
            # Print detailed results
            print(f"\n📸 {img_file}:")
            print(f"   Gender: {result['gender']['prediction']} (confidence: {result['gender']['confidence']:.3f})")
            print(f"   Gender probabilities:")
            for gender, prob in result['gender']['probabilities'].items():
                print(f"     {gender}: {prob:.3f}")
            print(f"   Identity: {result['identity']['prediction']} (confidence: {result['identity']['confidence']:.3f})")
            
        except Exception as e:
            print(f"❌ Error processing {img_file}: {str(e)}")
            continue
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    
    for result in results:
        print(f"{result['filename']:20} -> Gender: {result['gender_pred']:6} (conf: {result['gender_conf']:.3f})")
    
    return results

def analyze_gender_classification_issues(results):
    """Analyze potential issues with gender classification"""
    
    print("\n🔍 GENDER CLASSIFICATION ANALYSIS:")
    print("=" * 60)
    
    # Check for low confidence predictions
    low_confidence_threshold = 0.7
    low_conf_predictions = [r for r in results if r['gender_conf'] < low_confidence_threshold]
    
    if low_conf_predictions:
        print(f"\n⚠️  Low confidence predictions (< {low_confidence_threshold}):")
        for result in low_conf_predictions:
            print(f"   {result['filename']}: {result['gender_pred']} (conf: {result['gender_conf']:.3f})")
    
    # Check for close probability margins
    close_margin_threshold = 0.6  # If max prob < 0.6, it's quite uncertain
    uncertain_predictions = []
    
    for result in results:
        female_prob = result['gender_probs']['Female']
        male_prob = result['gender_probs']['Male']
        margin = abs(female_prob - male_prob)
        
        if margin < 0.2:  # Very close probabilities
            uncertain_predictions.append({
                'filename': result['filename'],
                'prediction': result['gender_pred'],
                'female_prob': female_prob,
                'male_prob': male_prob,
                'margin': margin
            })
    
    if uncertain_predictions:
        print(f"\n🤔 Uncertain predictions (close probabilities):")
        for pred in uncertain_predictions:
            print(f"   {pred['filename']}: {pred['prediction']} (F:{pred['female_prob']:.3f}, M:{pred['male_prob']:.3f}, margin:{pred['margin']:.3f})")
    
    return low_conf_predictions, uncertain_predictions

if __name__ == "__main__":
    print("🧪 ICAN Model Performance Test")
    print("=" * 60)
    
    # Test model with images
    results = test_model_with_images()
    
    if results:
        # Analyze issues
        low_conf, uncertain = analyze_gender_classification_issues(results)
        
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"   Total images tested: {len(results)}")
        print(f"   Low confidence predictions: {len(low_conf)}")
        print(f"   Uncertain predictions: {len(uncertain)}")
        
        if low_conf or uncertain:
            print(f"\n💡 RECOMMENDATIONS:")
            print(f"   1. Check image quality and preprocessing")
            print(f"   2. Consider retraining with more diverse data")
            print(f"   3. Adjust model architecture or hyperparameters")
            print(f"   4. Implement ensemble methods for better confidence")
    else:
        print("❌ No results to analyze") 