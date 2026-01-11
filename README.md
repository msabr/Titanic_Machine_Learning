# Titanic Survival Prediction

A comprehensive machine learning project predicting passenger survival on the Titanic using Logistic Regression.

## Overview

This project implements a systematic 15-step approach to solve the Titanic survival prediction problem, from data exploration to model deployment. The solution achieves ~80-82% accuracy with detailed analysis and visualizations.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running the Complete Solution

```bash
# Execute the comprehensive 15-step analysis
python titanic_solution.py
```

This will generate:
- `submission.csv` - Kaggle submission file
- `eda_visualizations.png` - EDA charts
- `roc_curve_baseline_model.png` - ROC curve for baseline model
- `roc_curve_optimized_model.png` - ROC curve for optimized model
- `feature_coefficients.png` - Feature importance visualization

### Alternative: Using Modular Approach

```bash
# Simple training and prediction using existing scripts
python main.py
```

## Project Structure

```
titanic_project/
├── data/
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessing.py
│   ├── features/
│   │   └── engineering.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── optimize.py
│   ├── visualization/
│   │   └── eda.py
│   └── utils/
│       └── config.py
├── main.py
└── requirements.txt
```

## 15-Step Methodology

1. **Data Loading & Exploration** - Load data, inspect structure, identify missing values
2. **Exploratory Data Analysis** - Analyze survival patterns by gender, class, age, etc.
3. **Missing Value Handling** - Impute Age, Embarked, Fare; create HasCabin feature
4. **Feature Engineering** - Create FamilySize, IsAlone, Title, AgeGroup, FareGroup
5. **Encoding** - One-Hot encode categorical variables with drop_first=True
6. **Feature Selection** - Remove irrelevant features (PassengerId, Name, Ticket, Cabin)
7. **Normalization** - Apply StandardScaler to all features
8. **Data Splitting** - 80/20 train/validation split with stratification
9. **Baseline Model** - Train initial Logistic Regression model
10. **Model Evaluation** - Accuracy, Confusion Matrix, Classification Report, ROC/AUC
11. **Hyperparameter Optimization** - GridSearchCV with C, penalty, solver
12. **Cross-Validation** - 5-fold StratifiedKFold validation
13. **Coefficient Analysis** - Interpret feature importance
14. **Test Predictions** - Generate submission file for Kaggle


## Key Results

- **Validation Accuracy**: ~80-82%
- **Cross-Validation Mean**: ~81-83%
- **ROC AUC Score**: ~0.84-0.86
- **Most Important Features**: Gender, Passenger Class, Title, Fare

## Key Findings

- Females had 74% survival rate vs. 19% for males
- 1st class passengers had 63% survival rate vs. 24% for 3rd class
- Children had higher survival rates
- Small families (2-4 members) had better survival chances
- Having cabin information correlated with survival

## Documentation

See [REPORT.md](REPORT.md) for a comprehensive analysis report including:
- Detailed methodology for each step
- Data insights and visualizations
- Model performance metrics
- Feature importance analysis
- Recommendations for improvement

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter

## License

MIT

