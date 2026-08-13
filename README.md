# Credit Card Fraud Detection

This project focuses on building an end-to-end machine learning pipeline for detecting fraudulent credit card transactions. The dataset is highly imbalanced making fraud detection a challenging classification problem.

The project covers data preparation, exploratory data analysis (EDA), preprocessing, model training, evaluation, and comparison of different machine learning models using appropriate fraud detection metrics.

# Hypotheses

## Before Training 

### 1. Which model do you expect to perform best for fraud detection? Why?

I expect Logistic Regression or Decision Tree to perform better than KNN.

Logistic Regression may perform well because the PCA transformed features can contain useful patterns for separating fraudulent and legitmare transactions.
I guess the PCA transformed features may benefit Logistic Regression by reducing multicollinearity between input features.This can make the estimation of model coefficients more stable.

Decision Tree may also perform well because it can capture non-linear relationship between features. For example a combination of several PCA transformed features may be useful for identifying fraud even if each feature alone is not sufficient.

KNN may be more affected by severe class imbalance. since approximately 99.83% of transactions belong to Class 0, most neighborhoods are excepted to contain far more legitmate transactions than fraudulent ones.So majority voting may favor Class 0 and lead to low fraud Recall.

### 2. Which metric is more important for this problem: Precision, Recall, or F1-score? Why?

Recall is particularly importand because missing a fraudulent transaction(False Negative) can be costly in a fraud detection system.However precision is also important because too many False positives can cause legitimate transactions to be incorrectly flagged.
Therefore i expect F1-score to be usefull for comparing models because it provides a balance between Precision and Recall.

### 3. What do you expect to happen if the model predicts all transactions as legitimate?

The model will achieve very high accuracy because legitimate transactions represents approximately 99.83% of the dataset.However it won't detect no fraudulent transactions.
Therefore ,Fraud Recall would be 0 and the model would be useless as a fraud detection system.

### 4. Do you expect feature scaling to significantly affect KNN performance?

Yes i expect feature scaling to significantly affect KNN performance beacause KNN relies on distance calculations.Features with larfer numerical scales such as Time and Amount can dominate the disatance calculation if the features are not scaled.Therefore I expect KNN with scaling to perform differntly, and potentially better, than KNN without scaling.

### 5. Do you expect the Decision Tree to overfit? Why?
Yes I expect an unrestricted or highly complex Decision Tree to overfit.
A decision Tree can continue splitting the training data into increasingly specific regions and may eventually learn noise and individuals characteristics of the training samples.Therefore i expect limiting parameters such as 'max_depth' to improve generalization.