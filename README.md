# House Prices - Advanced Regression Techniques

## Overview
This project is based on the Kaggle competition "[House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)." The goal is to predict the final sale price of homes based on a diverse set of features. 

## Competition Details
- **Competition URL**: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Objective**: Predict the sale price of homes in Ames, Iowa, using advanced regression techniques.
- **Evaluation Metric**: Root Mean Squared Logarithmic Error (RMSLE).

## Achievements
- **Public Leaderboard Score**: 0.1237 (RMSLE)
- **Position**: 530th

## Dataset
The dataset includes comprehensive information about houses, such as:
- **Features**: 79 explanatory variables (e.g., lot size, year built, etc.).
- **Target Variable**: SalePrice (continuous).

### Key Files
- **train.csv**: Training data with features and target variable.
- **test.csv**: Test data without the target variable.
- **sample_submission.csv**: Template for predictions.

## Approach

### Preprocessing
1. **Handling Missing Data**:
   - Replaced missing values with appropriate substitutes (mean, mode, or specific values like "None").
   - Dropped features with excessive missing values (>30%).

2. **Feature Engineering**:
   - Converted categorical variables to numeric using one-hot encoding.
   - Created new features by combining existing ones (e.g., TotalSF = 1stFlrSF + 2ndFlrSF + TotalBsmtSF).

3. **Scaling and Normalization**:
   - Applied logarithmic transformation to `SalePrice` to reduce skewness.
   - Scaled numerical features using StandardScaler.

### Modeling
1. **Algorithms Used**:
   - Gradient Boosting Machines (e.g., XGBoost, LightGBM, CatBoost).
   - Stacked models combining multiple algorithms for improved performance.
   
2. **Hyperparameter Tuning**:
   - Performed grid search and random search for optimal parameters.

3. **Validation**:
   - Used cross-validation (k-fold) to evaluate models and prevent overfitting.

### Final Submission
- Combined predictions from multiple models using weighted averaging to create the final submission file.

## Tools and Libraries
- **Python**: Programming language.
- **Jupyter Notebook**: Development environment.
- **Libraries**: pandas, numpy, scikit-learn, XGBoost, LightGBM, Matplotlib, Seaborn.

## Challenges
- Managing missing data effectively.
- Avoiding overfitting due to high-dimensional data.
- Balancing computational cost and model complexity during hyperparameter tuning.

## Results
Achieved a public leaderboard score of **0.1237 (RMSLE)**, securing the **530th position** out of thousands of participants. This reflects a competitive performance in a challenging problem domain.

## Future Improvements
- Experiment with advanced feature selection techniques to further improve model efficiency.
- Explore additional ensemble methods.
- Investigate the impact of external datasets to augment feature information.

## File Structure
- **New_price_prediction.ipynb**: Main notebook containing code for the project.
- **data/**: Folder containing datasets.
- **output/**: Folder for saving submission files and visualizations.

## How to Run
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Place the dataset files in the `data/` folder.
4. Run `New_price_prediction.ipynb` to reproduce the results.

## Contact
For any queries or collaboration, feel free to reach out:
- **Email**: [Your Email Addres](kaushalkathiriya1628@gmail.com)
- **Kaggle Profile**: [[Your Kaggle Profile Link](https://www.kaggle.com/kaushal0611)]

---
Thank you for exploring this project! 🎉
