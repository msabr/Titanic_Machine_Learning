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
    
    # Display recommendations
    display_completion_message()
    
    # Add recommendations
    print("\n" + "="*80)
    print("STEP 15: RECOMMENDATIONS FOR IMPROVEMENT")
    print("="*80)
    
    recommendations = """
    
1. ADVANCED FEATURE ENGINEERING
   • Interaction features: Pclass × Sex, Age × Pclass
   • Polynomial features for numerical variables
   • Extract more granular information from Cabin (deck letter)
   • Create fare-per-person feature (Fare / FamilySize)
   • Binning: Experiment with different age/fare grouping strategies
   
2. OUTLIER DETECTION AND TREATMENT
   • Identify outliers in Fare and Age using IQR or z-score
   • Analyze impact of extreme values on model performance
   • Consider robust scaling methods (RobustScaler)
   • Investigate passengers with very high fares or unusual ages
   
3. ALTERNATIVE ALGORITHMS
   • Random Forest: Handles non-linear relationships, feature interactions
   • Gradient Boosting (XGBoost, LightGBM, CatBoost): Often best performance
   • Support Vector Machines: Effective for binary classification
   • Neural Networks: Can capture complex patterns
   • Naive Bayes: Fast baseline, works well with categorical features
   
4. ENSEMBLE TECHNIQUES
   • Voting Classifier: Combine multiple models (Logistic, RF, XGBoost)
   • Stacking: Use meta-learner on predictions from base models
   • Bagging: Bootstrap aggregating for variance reduction
   • Boosting: Sequential learning to improve weak learners
   
5. FEATURE SELECTION TECHNIQUES
   • Recursive Feature Elimination (RFE)
   • SelectKBest with chi-squared or f_classif
   • L1 regularization for automatic feature selection
   • Principal Component Analysis (PCA) for dimensionality reduction
   
6. HYPERPARAMETER TUNING
   • Use RandomizedSearchCV for larger parameter spaces
   • Bayesian optimization (e.g., Optuna, Hyperopt)
   • Try different regularization strengths more granularly
   • Experiment with class_weight for imbalanced data handling
   
7. CROSS-VALIDATION STRATEGIES
   • Nested cross-validation for unbiased evaluation
   • Leave-one-out cross-validation for small datasets
   • Time-based splits if temporal patterns exist
   
8. DATA AUGMENTATION
   • SMOTE for balancing classes (if needed)
   • Generate synthetic samples using mixup
   
9. MODEL INTERPRETABILITY
   • SHAP values for feature importance
   • LIME for local interpretability
   • Partial dependence plots
   
10. ADDITIONAL DATA SOURCES
    • External datasets: Historical context, ship layout
    • Domain knowledge: Titanic disaster research
    • Feature engineering based on survival stories
        """
    
    print(recommendations)


if __name__ == "__main__":
    run_complete_analysis()