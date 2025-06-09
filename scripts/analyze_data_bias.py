"""
Analyze gender distribution and potential bias in training data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_gender_distribution():
    """Analyze gender distribution in training, validation, and test sets"""
    
    print("📊 GENDER DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Load datasets
    train_df = pd.read_csv('data/train_labels.csv')
    val_df = pd.read_csv('data/val_labels.csv')
    test_df = pd.read_csv('data/test_labels.csv')
    
    # Analyze training data
    print("\n🔍 TRAINING DATA:")
    train_gender_counts = train_df['gender'].value_counts()
    train_total = len(train_df)
    
    print(f"Total samples: {train_total}")
    for gender, count in train_gender_counts.items():
        percentage = (count / train_total) * 100
        print(f"{gender}: {count} ({percentage:.1f}%)")
    
    # Check if severely imbalanced
    gender_ratio = train_gender_counts['Female'] / train_gender_counts['Male']
    print(f"Female/Male ratio: {gender_ratio:.2f}")
    
    if gender_ratio > 2.0 or gender_ratio < 0.5:
        print("⚠️  WARNING: Severe gender imbalance detected!")
    
    # Analyze validation data
    print("\n🔍 VALIDATION DATA:")
    val_gender_counts = val_df['gender'].value_counts()
    val_total = len(val_df)
    
    print(f"Total samples: {val_total}")
    for gender, count in val_gender_counts.items():
        percentage = (count / val_total) * 100
        print(f"{gender}: {count} ({percentage:.1f}%)")
    
    # Analyze test data
    print("\n🔍 TEST DATA:")
    test_gender_counts = test_df['gender'].value_counts()
    test_total = len(test_df)
    
    print(f"Total samples: {test_total}")
    for gender, count in test_gender_counts.items():
        percentage = (count / test_total) * 100
        print(f"{gender}: {count} ({percentage:.1f}%)")
    
    # Per-person gender analysis
    print("\n👥 PER-PERSON GENDER ANALYSIS:")
    person_gender = train_df.groupby('person_id')['gender'].agg(['nunique', 'first']).reset_index()
    person_gender.columns = ['person_id', 'gender_variety', 'primary_gender']
    
    # Check for people with mixed gender labels (potential labeling errors)
    mixed_gender_people = person_gender[person_gender['gender_variety'] > 1]
    if len(mixed_gender_people) > 0:
        print(f"⚠️  FOUND {len(mixed_gender_people)} PEOPLE WITH MIXED GENDER LABELS:")
        for _, person in mixed_gender_people.iterrows():
            person_data = train_df[train_df['person_id'] == person['person_id']]
            print(f"  {person['person_id']}: {person_data['gender'].value_counts().to_dict()}")
        print("This suggests labeling inconsistencies!")
    else:
        print("✓ No mixed gender labels found (good!)")
    
    # Distribution by person
    gender_by_person = person_gender['primary_gender'].value_counts()
    print(f"\nUnique people by gender:")
    for gender, count in gender_by_person.items():
        percentage = (count / len(person_gender)) * 100
        print(f"{gender}: {count} people ({percentage:.1f}%)")
    
    return train_df, val_df, test_df, mixed_gender_people

def analyze_potential_model_bias():
    """Analyze potential sources of model bias"""
    
    print("\n🧠 POTENTIAL BIAS ANALYSIS")
    print("=" * 60)
    
    train_df, val_df, test_df, mixed_gender = analyze_gender_distribution()
    
    # Check class weights that might have been used
    train_gender_counts = train_df['gender'].value_counts()
    female_count = train_gender_counts.get('Female', 0)
    male_count = train_gender_counts.get('Male', 0)
    total_samples = female_count + male_count
    
    if female_count > male_count:
        dominant_class = 'Female'
        ratio = female_count / male_count if male_count > 0 else float('inf')
    else:
        dominant_class = 'Male'
        ratio = male_count / female_count if female_count > 0 else float('inf')
    
    print(f"\nClass imbalance analysis:")
    print(f"Dominant class: {dominant_class}")
    print(f"Imbalance ratio: {ratio:.2f}:1")
    
    if ratio > 1.5:
        print(f"⚠️  Significant class imbalance detected!")
        print(f"   Recommendation: Use class weighting or balanced sampling")
        
        # Calculate recommended class weights
        female_weight = total_samples / (2 * female_count) if female_count > 0 else 1.0
        male_weight = total_samples / (2 * male_count) if male_count > 0 else 1.0
        
        print(f"   Recommended class weights:")
        print(f"   Female: {female_weight:.3f}")
        print(f"   Male: {male_weight:.3f}")
    else:
        female_weight = 1.0
        male_weight = 1.0
    
    # Check for labeling errors
    if len(mixed_gender) > 0:
        print(f"\n⚠️  Labeling inconsistencies found!")
        print(f"   This could cause the model to learn incorrect patterns")
        print(f"   Recommendation: Clean the dataset labels")
    
    return {
        'female_weight': female_weight,
        'male_weight': male_weight,
        'has_labeling_errors': len(mixed_gender) > 0,
        'dominant_class': dominant_class,
        'imbalance_ratio': ratio
    }

if __name__ == "__main__":
    bias_info = analyze_potential_model_bias()
    
    print(f"\n💡 RECOMMENDATIONS FOR IMPROVING GENDER CLASSIFICATION:")
    print("=" * 60)
    print("1. Use class weights to handle imbalance")
    print("2. Clean labeling inconsistencies if any")
    print("3. Add more diverse training data")
    print("4. Consider data augmentation for minority class")
    print("5. Use techniques like focal loss for hard examples") 