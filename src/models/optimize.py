"""
Model optimization module.
"""

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression


class ModelOptimizer:
    """Handles model optimization."""
    
    def optimize_hyperparameters(self, X_train, y_train):
        """
        Optimize hyperparameters using GridSearchCV.
        """
        print("\n" + "="*80)
        print("STEP 11: HYPERPARAMETER OPTIMIZATION")
        print("="*80)
        
        print("\n--- Grid Search Configuration ---")
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        
        print(f"Parameters to optimize:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        
        print(f"\nCross-validation folds: 5")
        print(f"Scoring metric: accuracy")
        
        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            LogisticRegression(max_iter=1000, random_state=42),
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        print("\n--- Running Grid Search (this may take a moment) ---")
        grid_search.fit(X_train, y_train)
        
        # Best parameters
        print(f"\n--- Optimization Results ---")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Refit model with best parameters
        best_model = grid_search.best_estimator_
        
        print(f"\n✓ Model refitted with optimal hyperparameters")
        
        return best_model
    
    def cross_validate_model(self, best_model, baseline_model, X_train_full_scaled, y_train_full):
        """
        Validate optimized model using StratifiedKFold.
        """
        print("\n" + "="*80)
        print("STEP 12: CROSS-VALIDATION")
        print("="*80)
        
        print("\n--- Stratified K-Fold Cross-Validation ---")
        print("Configuration: 5 folds, stratified sampling")
        
        # Cross-validation with optimized model
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        cv_scores_optimized = cross_val_score(
            best_model, 
            X_train_full_scaled, 
            y_train_full,
            cv=skf,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Cross-validation with baseline model
        cv_scores_baseline = cross_val_score(
            LogisticRegression(max_iter=1000, random_state=42),
            X_train_full_scaled,
            y_train_full,
            cv=skf,
            scoring='accuracy',
            n_jobs=-1
        )
        
        print(f"\n--- Cross-Validation Results ---")
        print(f"\nOptimized Model:")
        print(f"  Mean Accuracy: {cv_scores_optimized.mean():.4f}")
        print(f"  Standard Deviation: {cv_scores_optimized.std():.4f}")
        print(f"  Scores by fold: {[f'{score:.4f}' for score in cv_scores_optimized]}")
        
        print(f"\nBaseline Model:")
        print(f"  Mean Accuracy: {cv_scores_baseline.mean():.4f}")
        print(f"  Standard Deviation: {cv_scores_baseline.std():.4f}")
        
        print(f"\n--- Comparison ---")
        improvement = cv_scores_optimized.mean() - cv_scores_baseline.mean()
        print(f"Improvement: {improvement:.4f} ({improvement*100:.2f}%)")
        
        if improvement > 0.01:
            print("✓ Hyperparameter optimization improved model performance")
        elif improvement > 0:
            print("✓ Slight improvement from optimization")
        else:
            print("⚠ No significant improvement (baseline already optimal)")
        
        return cv_scores_optimized