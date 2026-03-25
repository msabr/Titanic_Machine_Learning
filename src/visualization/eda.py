"""
Exploratory Data Analysis module.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.utils.config import AGE_BINS, AGE_LABELS


class EDAAnalyzer:
    """Performs comprehensive exploratory data analysis."""
    
    def exploratory_analysis(self, train_df):
        """
        Perform comprehensive exploratory data analysis.
        """
        print("\n" + "="*80)
        print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
        print("="*80)
        
        # Survival rate by Gender
        print("\n--- Survival Rate by Gender ---")
        gender_survival = train_df.groupby('Sex')['Survived'].agg(['mean', 'count'])
        print(gender_survival)
        
        # Survival rate by Class
        print("\n--- Survival Rate by Pclass ---")
        class_survival = train_df.groupby('Pclass')['Survived'].agg(['mean', 'count'])
        print(class_survival)
        
        # Survival rate by Embarked
        print("\n--- Survival Rate by Embarked ---")
        embarked_survival = train_df.groupby('Embarked')['Survived'].agg(['mean', 'count'])
        print(embarked_survival)
        
        # Survival rate by Age groups
        print("\n--- Survival Rate by Age Groups ---")
        train_df['AgeGroup_temp'] = pd.cut(train_df['Age'], 
                                           bins=AGE_BINS,
                                           labels=AGE_LABELS)
        age_survival = train_df.groupby('AgeGroup_temp')['Survived'].agg(['mean', 'count'])
        print(age_survival)
        
        # Family size analysis
        print("\n--- Survival Rate by Family Size ---")
        train_df['FamilySize_temp'] = train_df['SibSp'] + train_df['Parch'] + 1
        family_survival = train_df.groupby('FamilySize_temp')['Survived'].agg(['mean', 'count'])
        print(family_survival)
        
        # Correlation analysis
        print("\n--- Correlation Matrix for Numerical Variables ---")
        numerical_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        correlation_matrix = train_df[numerical_cols].corr()
        print(correlation_matrix['Survived'].sort_values(ascending=False))
        
        # Create visualizations
        self._create_eda_visualizations(train_df)
        
        # Conclusions
        print("\n--- Key Insights ---")
        print("1. Gender: Females have significantly higher survival rate (~74%) vs males (~19%)")
        print("2. Class: 1st class passengers had highest survival rate (~63%), 3rd class lowest (~24%)")
        print("3. Age: Children had higher survival rates")
        print("4. Family Size: Passengers with small families (2-4) had better survival rates")
        print("5. Fare: Higher fares correlate with better survival (proxy for class)")
        print("6. Most important features: Sex, Pclass, Fare, Age")
    
    def _create_eda_visualizations(self, train_df):
        """Create and save EDA visualizations."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Survival by Gender
        train_df.groupby('Sex')['Survived'].mean().plot(kind='bar', ax=axes[0, 0], color=['skyblue', 'salmon'])
        axes[0, 0].set_title('Survival Rate by Gender')
        axes[0, 0].set_ylabel('Survival Rate')
        axes[0, 0].set_xticklabels(['Female', 'Male'], rotation=0)
        
        # 2. Survival by Class
        train_df.groupby('Pclass')['Survived'].mean().plot(kind='bar', ax=axes[0, 1], color='green')
        axes[0, 1].set_title('Survival Rate by Passenger Class')
        axes[0, 1].set_ylabel('Survival Rate')
        axes[0, 1].set_xlabel('Pclass')
        
        # 3. Age distribution
        train_df[train_df['Survived']==1]['Age'].hist(ax=axes[0, 2], bins=30, alpha=0.5, label='Survived', color='green')
        train_df[train_df['Survived']==0]['Age'].hist(ax=axes[0, 2], bins=30, alpha=0.5, label='Died', color='red')
        axes[0, 2].set_title('Age Distribution by Survival')
        axes[0, 2].set_xlabel('Age')
        axes[0, 2].legend()
        
        # 4. Survival by Embarked
        train_df.groupby('Embarked')['Survived'].mean().plot(kind='bar', ax=axes[1, 0], color='purple')
        axes[1, 0].set_title('Survival Rate by Embarkation Port')
        axes[1, 0].set_ylabel('Survival Rate')
        axes[1, 0].set_xticklabels(['C', 'Q', 'S'], rotation=0)
        
        # 5. Survival by Family Size
        train_df.groupby('FamilySize_temp')['Survived'].mean().plot(kind='bar', ax=axes[1, 1], color='orange')
        axes[1, 1].set_title('Survival Rate by Family Size')
        axes[1, 1].set_ylabel('Survival Rate')
        axes[1, 1].set_xlabel('Family Size')
        
        # 6. Correlation heatmap
        numerical_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        sns.heatmap(train_df[numerical_cols].corr(), annot=True, cmap='coolwarm', ax=axes[1, 2], center=0)
        axes[1, 2].set_title('Correlation Heatmap')
        
        plt.tight_layout()
        plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
        print("\n[Visualization saved: eda_visualizations.png]")
        plt.close()