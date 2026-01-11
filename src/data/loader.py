"""
Data loading and initial exploration module.
"""

import pandas as pd


class DataLoader:
    """Handles data loading and initial exploration."""
    
    def __init__(self, train_path='data/train.csv', test_path='data/test.csv'):
        self.train_path = train_path
        self.test_path = test_path
        self.train_df = None
        self.test_df = None
    
    def load_and_explore(self):
        """
        Load datasets and perform initial exploration.
        """
        print("="*80)
        print("STEP 1: LOADING AND INITIAL EXPLORATION")
        print("="*80)
        
        # Load datasets
        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)
        
        print(f"\nTraining dataset shape: {self.train_df.shape}")
        print(f"Test dataset shape: {self.test_df.shape}")
        
        # Display first 5 rows
        print("\n--- First 5 rows of training data ---")
        print(self.train_df.head())
        
        # Show column information
        print("\n--- Column Information ---")
        print(self.train_df.info())
        
        # Descriptive statistics
        print("\n--- Descriptive Statistics for Numerical Variables ---")
        print(self.train_df.describe())
        
        # Identify missing values
        print("\n--- Missing Values Count ---")
        missing_train = self.train_df.isnull().sum()
        missing_test = self.test_df.isnull().sum()
        
        missing_df = pd.DataFrame({
            'Train': missing_train[missing_train > 0],
            'Test': missing_test[missing_test > 0]
        })
        print(missing_df)
        
        return self.train_df, self.test_df