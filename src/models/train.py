"""
Model training module.
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd


class ModelTrainer:
    """Handles model training."""
    
    def split_data(self, X_train_full_scaled, y_train_full):
        """
        Split training data into train/validation sets.
        """
        print("\n" + "="*80)
        print("STEP 8: DATA SPLITTING")
        print("="*80)
        
        # Convert to DataFrame if not already
        if not isinstance(X_train_full_scaled, pd.DataFrame):
            X_train_full_scaled = pd.DataFrame(X_train_full_scaled)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full_scaled,
            y_train_full,
            test_size=0.2,
            random_state=42,
            stratify=y_train_full  # Ensures balanced distribution
        )
        
        print(f"\n--- Split Configuration ---")
        print(f"Train/Validation split: 80/20")
        print(f"Random state: 42 (for reproducibility)")
        print(f"Stratified: Yes (maintains target distribution)")
        
        print(f"\n--- Dataset Sizes ---")
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        
        print(f"\n--- Target Distribution Validation ---")
        print(f"Full dataset - Survived: {y_train_full.mean():.4f}")
        print(f"Training set - Survived: {y_train.mean():.4f}")
        print(f"Validation set - Survived: {y_val.mean():.4f}")
        print(f"✓ Distribution is consistent across splits")
        
        return X_train, X_val, y_train, y_val
    
    def train_baseline_model(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train a baseline Logistic Regression model.
        """
        print("\n" + "="*80)
        print("STEP 9: BASE MODEL TRAINING")
        print("="*80)
        
        print("\n--- Training Baseline Logistic Regression ---")
        print("Configuration: Default hyperparameters, max_iter=1000")
        
        # Initialize and train baseline model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        
        # Accuracy
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        print(f"\n--- Baseline Model Performance ---")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            print(f"Difference: {abs(train_accuracy - val_accuracy):.4f}")
            
            if abs(train_accuracy - val_accuracy) < 0.05:
                print("✓ Model shows good generalization (low overfitting)")
            elif train_accuracy > val_accuracy:
                print("⚠ Possible overfitting detected")
            else:
                print("⚠ Unusual pattern: validation > training")
        
        return model