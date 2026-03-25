"""
Model evaluation module.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                            roc_curve, auc, roc_auc_score)


class ModelEvaluator:
    """Handles model evaluation."""
    
    def evaluate_model(self, model, X_train, X_val, y_train, y_val, model_name="Model"):
        """
        Comprehensive model evaluation.
        """
        print("\n" + "="*80)
        print(f"STEP 10: MODEL EVALUATION - {model_name}")
        print("="*80)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Accuracy
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        print(f"\n--- Accuracy Scores ---")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        
        # Confusion Matrix
        print(f"\n--- Confusion Matrix (Validation Set) ---")
        cm = confusion_matrix(y_val, y_val_pred)
        print(cm)
        print(f"\nInterpretation:")
        print(f"  True Negatives (TN): {cm[0,0]} | False Positives (FP): {cm[0,1]}")
        print(f"  False Negatives (FN): {cm[1,0]} | True Positives (TP): {cm[1,1]}")
        
        # Classification Report
        print(f"\n--- Classification Report (Validation Set) ---")
        print(classification_report(y_val, y_val_pred, 
                                   target_names=['Died', 'Survived']))
        
        # ROC Curve and AUC
        fpr, tpr, thresholds = roc_curve(y_val, y_val_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        print(f"\n--- ROC AUC Score ---")
        print(f"AUC: {roc_auc:.4f}")
        
        # Plot ROC Curve
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(f'roc_curve_{model_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
        print(f"\n[Visualization saved: roc_curve_{model_name.replace(' ', '_').lower()}.png]")
        plt.close()
        
        # Overfitting/Underfitting Analysis
        print(f"\n--- Overfitting/Underfitting Analysis ---")
        diff = train_acc - val_acc
        if diff < 0.02:
            print("✓ Well-fitted model (minimal overfitting)")
        elif diff < 0.05:
            print("⚠ Slight overfitting detected")
        else:
            print("⚠⚠ Significant overfitting - consider regularization")
        
        if val_acc < 0.70:
            print("⚠ Low validation accuracy - possible underfitting")
        
        return val_acc, roc_auc
    
    def analyze_coefficients(self, model, feature_names):
        """
        Interpret regression coefficients.
        """
        print("\n" + "="*80)
        print("STEP 13: COEFFICIENT ANALYSIS")
        print("="*80)
        
        # Get coefficients
        coefficients = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_[0]
        })
        
        # Sort by absolute value
        coefficients['Abs_Coefficient'] = coefficients['Coefficient'].abs()
        coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)
        
        print("\n--- Top 10 Most Impactful Features ---")
        print(coefficients[['Feature', 'Coefficient']].head(10).to_string(index=False))
        
        print("\n--- Interpretation ---")
        print("\nPositive coefficients increase survival probability:")
        positive_coef = coefficients[coefficients['Coefficient'] > 0].head(5)
        for idx, row in positive_coef.iterrows():
            print(f"  • {row['Feature']}: {row['Coefficient']:.4f}")
        
        print("\nNegative coefficients decrease survival probability:")
        negative_coef = coefficients[coefficients['Coefficient'] < 0].head(5)
        for idx, row in negative_coef.iterrows():
            print(f"  • {row['Feature']}: {row['Coefficient']:.4f}")
        
        # Visualization
        plt.figure(figsize=(12, 8))
        top_features = coefficients.head(15)
        colors = ['green' if c > 0 else 'red' for c in top_features['Coefficient']]
        plt.barh(range(len(top_features)), top_features['Coefficient'], color=colors, alpha=0.7)
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Coefficient Value')
        plt.title('Top 15 Feature Coefficients (Logistic Regression)')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        plt.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('feature_coefficients.png', dpi=300, bbox_inches='tight')
        print("\n[Visualization saved: feature_coefficients.png]")
        plt.close()
        
        return coefficients
    
    def generate_predictions(self, model, X_test_full_scaled, test_passenger_ids):
        """
        Apply preprocessing and model to test set.
        """
        print("\n" + "="*80)
        print("STEP 14: TEST SET PREDICTIONS")
        print("="*80)
        
        # Verify no missing values
        print("\n--- Verification: Missing Values in Test Set ---")
        missing_test = X_test_full_scaled.isnull().sum().sum()
        print(f"Total missing values: {missing_test}")
        
        if missing_test > 0:
            print("⚠ Missing values found - this should not happen after proper preprocessing")
            print("Missing values by column:")
            print(X_test_full_scaled.isnull().sum()[X_test_full_scaled.isnull().sum() > 0])
            raise ValueError("Missing values detected in test set after preprocessing")
        
        # Generate predictions using optimized model
        print("\n--- Generating Predictions ---")
        predictions = model.predict(X_test_full_scaled)
        
        # Create submission file
        submission = pd.DataFrame({
            'PassengerId': test_passenger_ids,
            'Survived': predictions.astype(int)
        })
        
        print(f"✓ Predictions generated for {len(predictions)} passengers")
        
        print(f"\n--- Prediction Statistics ---")
        print(f"Predicted Survived: {predictions.sum()} ({predictions.sum()/len(predictions)*100:.2f}%)")
        print(f"Predicted Died: {len(predictions) - predictions.sum()} ({(len(predictions) - predictions.sum())/len(predictions)*100:.2f}%)")
        
        print("\n--- Sample Predictions ---")
        print(submission.head(10))
        
        return submission