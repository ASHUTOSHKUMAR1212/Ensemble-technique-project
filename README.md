# Ensemble-technique-project

Project Objective

As a data scientist, the objective is to build machine learning models to predict customers likely to churn. By analyzing customer behavior and identifying key factors contributing to churn, we aim to improve the company's retention efforts.
Steps and Tasks
1. Data Understanding & Exploration

    Load two CSV files into Pandas DataFrames.
    Merge the DataFrames on customerID and verify that all columns are included.

2. Data Cleaning & Analysis

    Impute missing and unexpected values.
    Ensure continuous variables like MonthlyCharges and TotalCharges are of the float data type.
    Generate pie charts for categorical variables to visualize distribution.
    Encode categorical variables using appropriate techniques.
    Split data into an 80/20 train-test ratio.
    Standardize the data for machine learning.

3. Model Building & Performance Improvement

Models to be trained and optimized:

    Decision Tree:
        Train a base model and evaluate performance.
        Use GridSearchCV to tune the model and improve performance.

    Random Forest:
        Train a base model and evaluate performance.
        Use GridSearchCV to improve performance.

    AdaBoost:
        Train a base model and evaluate performance.
        Tune the model using GridSearchCV.

    GradientBoost:
        Train a base model and evaluate performance.
        Use GridSearchCV for hyperparameter tuning.

4. Model Comparison & Final Conclusion

    Compare the performance of all models during both the training and test stages.
    Analyze which model performed the best and why.
    Provide final conclusions from the model evaluation.

Performance Metrics

For each model, the following metrics will be evaluated:

    Accuracy
    Precision
    Recall
    F1 Score
    ROC-AUC Score

Tools and Libraries Used

    Python: Programming language used for model building and evaluation.
    Pandas: Data manipulation and cleaning.
    NumPy: Numerical operations.
    Matplotlib/Seaborn: Data visualization.
    Scikit-learn: Machine learning library used for training models and performing hyperparameter tuning.

Conclusion

The project will conclude with a detailed analysis of which model performs best for predicting customer churn. The model with the highest accuracy, recall, and AUC-ROC score on the test dataset will be recommended for deployment.
