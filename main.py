"""
Titanic Survival Prediction - Complete Solution
Main entry point for the modularized Titanic project.
"""

import warnings
warnings.filterwarnings('ignore')

from src.utils.config import setup_environment
from src.data.loader import DataLoader
from src.visualization.eda import EDAAnalyzer
from src.data.preprocessing import DataPreprocessor
from src.features.engineering import FeatureEngineer
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator
from src.models.optimize import ModelOptimizer
from src.utils.config import save_predictions, display_completion_message
import pandas as pd
from sklearn.preprocessing import StandardScaler


def run_complete_analysis():
    """Execute all 15 steps in sequence."""
    print("\n" + "="*80)
    print("TITANIC SURVIVAL PREDICTION - COMPLETE SOLUTION")
    print("Systematic Approach using Logistic Regression")
    print("="*80)
    
    # Setup environment
    setup_environment()
    
    # Initialize components
    loader = DataLoader(train_path='data/train.csv', test_path='data/test.csv')
    eda_analyzer = EDAAnalyzer()
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    optimizer = ModelOptimizer()
    
    # STEP 1: Load data
    train_df, test_df = loader.load_and_explore()
    
    # STEP 2: EDA
    eda_analyzer.exploratory_analysis(train_df)
    
    # STEP 3: Handle missing values
    train_df, test_df = preprocessor.handle_missing_values(train_df, test_df)
    
    # STEP 4: Feature engineering
    train_df, test_df, fare_bins = feature_engineer.feature_engineering(train_df, test_df)
    
    # STEP 5: Encoding
    train_df, test_df = feature_engineer.encode_features(train_df, test_df)
    
    # STEP 6: Feature selection
    X_train_full, y_train_full, X_test_full, test_passenger_ids, feature_names = feature_engineer.select_features(train_df, test_df)
    
    # STEP 7: Normalization
    scaler = StandardScaler()
    X_train_full_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_full),
        columns=X_train_full.columns,
        index=X_train_full.index
    )
    
    X_test_full_scaled = pd.DataFrame(
        scaler.transform(X_test_full),
        columns=X_test_full.columns,
        index=X_test_full.index
    )
    
    print(f"\n--- Sample Statistics After Scaling ---")
    print(f"Mean: {X_train_full_scaled.mean().mean():.6f} (should be ≈ 0)")
    print(f"Std: {X_train_full_scaled.std().mean():.6f} (should be ≈ 1)")
    
    # STEP 8: Data splitting
    X_train, X_val, y_train, y_val = trainer.split_data(X_train_full_scaled, y_train_full)
    
    # STEP 9: Train baseline model
    baseline_model = trainer.train_baseline_model(X_train, y_train, X_val, y_val)
    
    # STEP 10: Evaluate baseline model
    evaluator.evaluate_model(baseline_model, X_train, X_val, y_train, y_val, "Baseline Model")
    
    # STEP 11: Hyperparameter optimization
    best_model = optimizer.optimize_hyperparameters(X_train, y_train)
    
    # STEP 12: Evaluate optimized model
    evaluator.evaluate_model(best_model, X_train, X_val, y_train, y_val, "Optimized Model")
    
    # STEP 13: Cross-validation
    optimizer.cross_validate_model(best_model, baseline_model, X_train_full_scaled, y_train_full)
    
    # STEP 14: Coefficient analysis
    coefficients = evaluator.analyze_coefficients(best_model, feature_names)
    
    # STEP 15: Generate predictions
    predictions = evaluator.generate_predictions(best_model, X_test_full_scaled, test_passenger_ids)
    save_predictions(predictions)
    
    
    
    

if __name__ == "__main__":
    run_complete_analysis()