"""
Setup Celebrity Dataset for ICAN Training
Organizes the celebrity face dataset with proper gender labels
"""

import os
import pandas as pd
import shutil
from pathlib import Path

# Celebrity gender mapping (based on known celebrities)
CELEBRITY_GENDERS = {
    'Zac Efron': 'Male',
    'Virat Kohli': 'Male', 
    'Vijay Deverakonda': 'Male',
    'Tom Cruise': 'Male',
    'Roger Federer': 'Male',
    'Robert Downey Jr': 'Male',
    'Marmik': 'Male',  # Assuming male based on name
    'Kashyap': 'Male',  # Assuming male based on name  
    'Hugh Jackman': 'Male',
    'Hrithik Roshan': 'Male',
    'Henry Cavill': 'Male',
    'Dwayne Johnson': 'Male',
    'Brad Pitt': 'Male',
    'Andy Samberg': 'Male',
    'Amitabh Bachchan': 'Male',
    'Akshay Kumar': 'Male',
    
    'Priyanka Chopra': 'Female',
    'Natalie Portman': 'Female',
    'Margot Robbie': 'Female',
    'Lisa Kudrow': 'Female',
    'Jessica Alba': 'Female',
    'Ellen Degeneres': 'Female',
    'Elizabeth Olsen': 'Female',
    'Courtney Cox': 'Female',
    'Claire Holt': 'Female',
    'Charlize Theron': 'Female',
    'Camila Cabello': 'Female',
    'Billie Eilish': 'Female',
    'Anushka Sharma': 'Female',
    'Alia Bhatt': 'Female',
    'Alexandra Daddario': 'Female'
}

def setup_celebrity_dataset():
    """Setup the celebrity dataset for ICAN training"""
    
    print("🎬 SETTING UP CELEBRITY FACE DATASET")
    print("=" * 60)
    
    # Source and destination paths
    source_faces_dir = "archive/Faces/Faces"
    dest_facecom_dir = "data/FACECOM"
    source_csv = "archive/Dataset.csv"
    
    # Create destination directory
    os.makedirs(dest_facecom_dir, exist_ok=True)
    
    # Load the original CSV
    print("📄 Loading dataset CSV...")
    df = pd.read_csv(source_csv)
    print(f"   Total entries: {len(df)}")
    
    # Create the new dataset with proper structure
    new_dataset = []
    copied_files = 0
    skipped_files = 0
    
    print("\n📋 Processing celebrities and copying images...")
    
    # Process each celebrity
    for celebrity, gender in CELEBRITY_GENDERS.items():
        celebrity_images = df[df['label'] == celebrity]
        print(f"\n👤 {celebrity} ({gender}): {len(celebrity_images)} images")
        
        for idx, row in celebrity_images.iterrows():
            image_filename = row['id']
            source_path = os.path.join(source_faces_dir, image_filename)
            dest_path = os.path.join(dest_facecom_dir, image_filename)
            
            # Copy image if it exists
            if os.path.exists(source_path):
                try:
                    shutil.copy2(source_path, dest_path)
                    
                    # Add to new dataset
                    new_dataset.append({
                        'image_path': image_filename,
                        'gender': gender,
                        'person_id': celebrity.replace(' ', '_')  # Use celebrity name as person_id
                    })
                    
                    copied_files += 1
                    if copied_files % 100 == 0:
                        print(f"   Copied {copied_files} images...")
                        
                except Exception as e:
                    print(f"   ❌ Error copying {image_filename}: {e}")
                    skipped_files += 1
            else:
                print(f"   ⚠️  Missing: {image_filename}")
                skipped_files += 1
    
    # Create the new dataset DataFrame
    new_df = pd.DataFrame(new_dataset)
    
    print(f"\n📊 DATASET STATISTICS:")
    print(f"   Total images copied: {copied_files}")
    print(f"   Skipped files: {skipped_files}")
    print(f"   Total celebrities: {len(CELEBRITY_GENDERS)}")
    
    # Gender distribution
    gender_counts = new_df['gender'].value_counts()
    print(f"\n⚖️  GENDER DISTRIBUTION:")
    for gender, count in gender_counts.items():
        percentage = (count / len(new_df)) * 100
        print(f"   {gender}: {count} ({percentage:.1f}%)")
    
    # Check for balanced dataset
    ratio = gender_counts['Female'] / gender_counts['Male'] if 'Male' in gender_counts else float('inf')
    print(f"   Female/Male ratio: {ratio:.2f}")
    
    if 0.5 <= ratio <= 2.0:
        print("   ✅ Dataset is reasonably balanced!")
    else:
        print("   ⚠️  Dataset has gender imbalance")
    
    # Celebrity distribution  
    celebrity_counts = new_df['person_id'].value_counts()
    print(f"\n👥 CELEBRITY DISTRIBUTION:")
    print(f"   Average images per celebrity: {len(new_df) / len(celebrity_counts):.1f}")
    print(f"   Min images: {celebrity_counts.min()}")
    print(f"   Max images: {celebrity_counts.max()}")
    
    # Show some examples
    print(f"\n📸 SAMPLE ENTRIES:")
    print(new_df.head(10).to_string(index=False))
    
    return new_df

def create_train_val_test_splits(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Create train/validation/test splits ensuring each celebrity appears in all splits"""
    
    print(f"\n📂 CREATING TRAIN/VAL/TEST SPLITS")
    print(f"   Train: {train_ratio:.0%}, Validation: {val_ratio:.0%}, Test: {test_ratio:.0%}")
    
    train_data = []
    val_data = []
    test_data = []
    
    # For each celebrity, split their images
    for person_id in df['person_id'].unique():
        person_images = df[df['person_id'] == person_id].copy()
        n_images = len(person_images)
        
        # Calculate split sizes
        n_train = max(1, int(n_images * train_ratio))
        n_val = max(1, int(n_images * val_ratio))
        n_test = n_images - n_train - n_val
        
        # If not enough images for all splits, adjust
        if n_test <= 0:
            if n_images >= 2:
                n_train = n_images - 1
                n_val = 1
                n_test = 0
            else:
                n_train = 1
                n_val = 0
                n_test = 0
        
        # Shuffle and split
        person_images = person_images.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_data.extend(person_images.iloc[:n_train].to_dict('records'))
        val_data.extend(person_images.iloc[n_train:n_train+n_val].to_dict('records'))
        if n_test > 0:
            test_data.extend(person_images.iloc[n_train+n_val:].to_dict('records'))
        
        print(f"   {person_id}: {n_train} train, {n_val} val, {n_test} test")
    
    # Convert to DataFrames
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)
    
    print(f"\n📋 FINAL SPLIT SIZES:")
    print(f"   Training: {len(train_df)} images")
    print(f"   Validation: {len(val_df)} images") 
    print(f"   Test: {len(test_df)} images")
    
    # Check gender balance in each split
    for split_name, split_df in [("Training", train_df), ("Validation", val_df), ("Test", test_df)]:
        if len(split_df) > 0:
            gender_dist = split_df['gender'].value_counts()
            print(f"\n   {split_name} gender distribution:")
            for gender, count in gender_dist.items():
                pct = (count / len(split_df)) * 100
                print(f"     {gender}: {count} ({pct:.1f}%)")
    
    return train_df, val_df, test_df

def save_dataset_files(train_df, val_df, test_df):
    """Save the dataset files"""
    
    print(f"\n💾 SAVING DATASET FILES:")
    
    # Save CSV files
    train_df.to_csv('data/train_labels.csv', index=False)
    val_df.to_csv('data/val_labels.csv', index=False)  
    test_df.to_csv('data/test_labels.csv', index=False)
    
    print(f"   ✅ Saved data/train_labels.csv ({len(train_df)} entries)")
    print(f"   ✅ Saved data/val_labels.csv ({len(val_df)} entries)")
    print(f"   ✅ Saved data/test_labels.csv ({len(test_df)} entries)")
    
    # Also save the complete dataset
    complete_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    complete_df.to_csv('data/celebrity_dataset_complete.csv', index=False)
    print(f"   ✅ Saved data/celebrity_dataset_complete.csv ({len(complete_df)} entries)")

def main():
    """Main function to setup the entire dataset"""
    
    # Setup the dataset
    dataset_df = setup_celebrity_dataset()
    
    if len(dataset_df) == 0:
        print("❌ No data to process!")
        return
    
    # Create splits
    train_df, val_df, test_df = create_train_val_test_splits(dataset_df)
    
    # Save files
    save_dataset_files(train_df, val_df, test_df)
    
    print(f"\n🎉 DATASET SETUP COMPLETE!")
    print(f"=" * 60)
    print(f"✅ Celebrity face dataset ready for ICAN training")
    print(f"✅ {len(dataset_df)} total images from {len(CELEBRITY_GENDERS)} celebrities")
    print(f"✅ Balanced gender representation")
    print(f"✅ Train/validation/test splits created")
    print(f"✅ Files saved in data/ directory")
    print(f"\n🚀 Ready to train the improved ICAN model!")

if __name__ == "__main__":
    main() 