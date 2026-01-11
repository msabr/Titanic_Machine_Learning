"""
Feature engineering module.
"""

import pandas as pd
import numpy as np
from src.utils.config import AGE_BINS, AGE_LABELS


class FeatureEngineer:
    """Handles feature engineering and encoding."""
    
    def __init__(self):
        self.fare_bins = None
        self.feature_names = None
    
    def feature_engineering(self, train_df, test_df):
        """
        Create new features to improve model performance.
        """
        print("\n" + "="*80)
        print("STEP 4: FEATURE ENGINEERING")
        print("="*80)
        
        for idx, df in enumerate([train_df, test_df]):
            # FamilySize
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            if idx == 0:
                print(f"\n✓ Created FamilySize: SibSp + Parch + 1")
            
            # IsAlone
            df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
            if idx == 0:
                print(f"✓ Created IsAlone: 1 if FamilySize == 1, else 0")
            
            # Title extraction
            df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
            
            # Group rare titles
            title_mapping = {
                'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
                'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
                'Mlle': 'Miss', 'Mme': 'Mrs', 'Don': 'Rare', 'Dona': 'Rare',
                'Lady': 'Rare', 'Countess': 'Rare', 'Jonkheer': 'Rare', 'Sir': 'Rare',
                'Capt': 'Rare', 'Ms': 'Miss'
            }
            df['Title'] = df['Title'].map(title_mapping)
            df['Title'].fillna('Rare', inplace=True)
            if idx == 0:
                print(f"✓ Created Title: Extracted from Name (Mr, Mrs, Miss, Master, Rare)")
            
            # AgeGroup - use consistent bins
            df['AgeGroup'] = pd.cut(df['Age'], 
                                    bins=AGE_BINS,
                                    labels=AGE_LABELS)
            if idx == 0:
                print(f"✓ Created AgeGroup: Categorized Age (Child, Adolescent, Adult, Middle-aged, Senior)")
            
            # FareGroup - calculate bins from training data only
            if idx == 0:
                # First iteration - training data
                df['FareGroup'], self.fare_bins = pd.qcut(df['Fare'], q=4, 
                                                          labels=['Low', 'Medium', 'High', 'Very High'], 
                                                          duplicates='drop', retbins=True)
                print(f"✓ Created FareGroup: Quartiles (Low, Medium, High, Very High)")
            else:
                # Second iteration - test data, use training bins
                df['FareGroup'] = pd.cut(df['Fare'], bins=self.fare_bins,
                                        labels=['Low', 'Medium', 'High', 'Very High'],
                                        include_lowest=True)
            
            # HasCabin
            df['HasCabin'] = df['Cabin'].notna().astype(int)
            if idx == 0:
                print(f"✓ Created HasCabin: 1 if Cabin not empty, else 0")
        
        # Display new features
        print("\n--- Sample of Engineered Features ---")
        print(train_df[['FamilySize', 'IsAlone', 'Title', 'AgeGroup', 'FareGroup', 'HasCabin']].head(10))
        
        return train_df, test_df, self.fare_bins
    
    def encode_features(self, train_df, test_df):
        """
        Encode categorical variables appropriately.
        """
        print("\n" + "="*80)
        print("STEP 5: ENCODING CATEGORICAL VARIABLES")
        print("="*80)
        
        # Combine train and test for consistent encoding
        train_len = len(train_df)
        combined = pd.concat([train_df, test_df], axis=0, sort=False)
        
        # One-Hot Encoding with drop_first=True to avoid multicollinearity
        categorical_features = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup']
        
        print(f"\n--- Applying One-Hot Encoding ---")
        print(f"Features: {categorical_features}")
        print(f"Setting drop_first=True to avoid multicollinearity")
        
        combined = pd.get_dummies(combined, columns=categorical_features, drop_first=True)
        
        # Split back to train and test
        train_df = combined[:train_len]
        test_df = combined[train_len:]
        
        print(f"\n✓ Encoding completed")
        print(f"Train shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")
        
        # Display encoded columns
        print("\n--- Encoded Columns (sample) ---")
        encoded_cols = [col for col in train_df.columns if any(cat in col for cat in categorical_features)]
        print(encoded_cols[:10])
        
        return train_df, test_df
    
    def select_features(self, train_df, test_df):
        """
        Select relevant features for modeling.
        """
        print("\n" + "="*80)
        print("STEP 6: FEATURE SELECTION")
        print("="*80)
        
        # Columns to drop
        cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 
                       'AgeGroup_temp', 'FamilySize_temp']
        
        # Drop from train (keep Survived)
        cols_to_drop_train = [col for col in cols_to_drop if col in train_df.columns]
        if 'Survived' in train_df.columns:
            X_train_full = train_df.drop(cols_to_drop_train + ['Survived'], axis=1, errors='ignore')
            y_train_full = train_df['Survived']
        
        # Drop from test (save PassengerId for submission)
        test_passenger_ids = test_df['PassengerId'].copy()
        cols_to_drop_test = [col for col in cols_to_drop if col in test_df.columns]
        # Also drop Survived if it exists in test (from concatenation)
        cols_to_drop_test_all = cols_to_drop_test + ['Survived']
        X_test_full = test_df.drop(cols_to_drop_test_all, axis=1, errors='ignore')
        
        print(f"\n--- Excluded Features ---")
        print(f"Dropped: {cols_to_drop_train}")
        
        print(f"\n--- Final Feature Set ---")
        self.feature_names = X_train_full.columns.tolist()
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Features: {self.feature_names}")
        
        return X_train_full, y_train_full, X_test_full, test_passenger_ids, self.feature_names