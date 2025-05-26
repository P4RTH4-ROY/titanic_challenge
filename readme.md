# Titanic Survival Prediction Challenge

This repository contains my solution for the [Kaggle Titanic Machine Learning Challenge](https://www.kaggle.com/c/titanic), implemented in Google Colab. The goal is to predict which passengers survived the Titanic shipwreck using machine learning.

## Dataset Overview

The Titanic dataset contains information about 891 passengers aboard the RMS Titanic, including:

- **Demographic features**: Passenger class, Name, Sex, Age, SibSp (siblings/spouses aboard), Parch (parents/children aboard)
- **Ticket information**: Ticket number, Fare, Cabin, Embarked port
- **Target variable**: Survival (0 = No, 1 = Yes)

Key characteristics:
- Contains missing values (particularly in Age and Cabin columns)
- Mix of numerical and categorical features
- Imbalanced classes (about 38% survived in training set)

## Implementation

### Approach
- Performed exploratory data analysis (EDA) to understand feature distributions and relationships
- Engineered new features from existing ones
- Handled missing values and categorical variables
- Trained a Random Forest classifier

### Results
- Achieved **75% accuracy** on the test set using Random Forest
- Feature importance analysis revealed that passenger class, sex, and fare were the most predictive features

## Requirements
The notebook was developed in Google Colab with:
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib/seaborn (for visualization)

## How to Use
1. Open the notebook in Google Colab
2. Run cells sequentially
3. The notebook includes comments explaining each step

## Future Improvements
- Experiment with other models (e.g., Gradient Boosting, Neural Networks)
- More sophisticated feature engineering
- Hyperparameter tuning for better performance
- Ensemble methods

## Acknowledgments
- Kaggle for hosting the competition
- Titanic dataset provided by Kaggle
