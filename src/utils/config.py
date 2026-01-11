"""
Configuration and utility functions for the Titanic project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def setup_environment():
    """Set up the visualization environment."""
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)


def save_predictions(submission):
    """Save predictions to CSV file."""
    submission.to_csv('submission.csv', index=False)
    print(f"✓ Submission file saved: submission.csv")


def display_completion_message():
    """Display completion message with generated files."""
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    print("  • submission.csv - Kaggle submission file")
    print("  • eda_visualizations.png - EDA charts")
    print("  • roc_curve_baseline_model.png - ROC curve for baseline")
    print("  • roc_curve_optimized_model.png - ROC curve for optimized model")
    print("  • feature_coefficients.png - Feature importance visualization")
    print("\n" + "="*80)


# Constants for feature engineering
AGE_BINS = [0, 12, 18, 35, 60, 100]
AGE_LABELS = ['Child', 'Adolescent', 'Adult', 'Middle-aged', 'Senior']