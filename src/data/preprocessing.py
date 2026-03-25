"""
Data preprocessing module.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """Handles data preprocessing tasks."""
    
    def handle_missing_values(self, train_df, test_df):
        """
        Impute missing values with appropriate strategies.
        """
        print("\n" + "="*80)
        print("STEP 3: HANDLING MISSING VALUES")
        print("="*80)
        
        # Age: Use median by group (Pclass, Sex)
        print("\n--- Imputing Age ---")
        print("Strategy: Median by group (Pclass, Sex)")
        print("Justification: Age varies significantly by class and gender")
        
        # Calculate median age by Pclass and Sex from TRAINING data only
        age_median_by_group = train_df.groupby(['Pclass', 'Sex'])['Age'].median()
        overall_median_age = train_df['Age'].median()
        
        # Apply to both train and test using the same medians
        for df in [train_df, test_df]:
            # Fill missing ages based on Pclass and Sex
            for (pclass, sex), median_age in age_median_by_group.items():
                mask = (df['Pclass'] == pclass) & (df['Sex'] == sex) & (df['Age'].isnull())
                df.loc[mask, 'Age'] = median_age
            
            # If any Age still missing (edge case), fill with overall median from training
            df['Age'].fillna(overall_median_age, inplace=True)
        
        # Embarked: Use mode
        print("\n--- Imputing Embarked ---")
        print("Strategy: Mode (most common value)")
        mode_embarked = train_df['Embarked'].mode()[0]
        print(f"Justification: Only 2 missing values, using mode: {mode_embarked}")
        
        train_df['Embarked'].fillna(mode_embarked, inplace=True)
        test_df['Embarked'].fillna(mode_embarked, inplace=True)
        
        # Cabin: Create HasCabin feature (done in feature engineering)
        print("\n--- Handling Cabin ---")
        print("Strategy: Create binary feature 'HasCabin'")
        print("Justification: Cabin number itself not useful, but having cabin info indicates deck/status")
        
        # Fare: Use median (for test set)
        print("\n--- Imputing Fare ---")
        print("Strategy: Median (from training set)")
        fare_median = train_df['Fare'].median()
        # Only fill test set - train set should not have missing Fare
        test_df['Fare'].fillna(fare_median, inplace=True)
        print(f"Justification: Uses training set median for test imputation: {fare_median:.2f}")
        
        # Verify no missing values in key columns
        print("\n--- Verification: Missing Values After Imputation ---")
        print("Train:", train_df[['Age', 'Embarked', 'Fare']].isnull().sum())
        print("Test:", test_df[['Age', 'Embarked', 'Fare']].isnull().sum())
        
        return train_df, test_df
    
    def normalize_features(self, X_train_full, X_test_full):
        """
        Apply StandardScaler to numerical features.
        """
        print("\n" + "="*80)
        print("STEP 7: NORMALIZATION/STANDARDIZATION")
        print("="*80)
        
        print("\n--- Rationale ---")
        print("• Logistic Regression uses gradient descent, which converges faster with normalized features")
        print("• Features with larger scales can dominate the optimization process")
        print("• StandardScaler ensures all features have mean=0 and std=1")
        print("• Regularization (L1/L2) works better when features are on same scale")
        
        # Initialize scaler
        scaler = StandardScaler()
        
        # Fit on training data
        X_train_full_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_full),
            columns=X_train_full.columns,
            index=X_train_full.index
        )
        
        # Transform test data
        X_test_full_scaled = pd.DataFrame(
            scaler.transform(X_test_full),
            columns=X_test_full.columns,
            index=X_test_full.index
        )
        
        print(f"\n✓ StandardScaler fitted on training data")
        print(f"✓ Applied to both training and test sets")
        print(f"\n--- Sample Statistics After Scaling ---")
        print(f"Mean: {X_train_full_scaled.mean().mean():.6f} (should be ≈ 0)")
        print(f"Std: {X_train_full_scaled.std().mean():.6f} (should be ≈ 1)")
        
        return X_train_full_scaled, X_test_full_scaled, scaler