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

---

## After Training 

### 1. Was your initial hypothesis correct?

**Partially.**

Our initial hypothesis was that **Logistic Regression or Decision Tree would outperform KNN**, while KNN would be negatively affected by the severe class imbalance.

The results showed that this hypothesis was **not completely correct**. Scaled KNN with `k=5` achieved the best overall balance among the evaluated models, especially after threshold adjustment.

However, our hypothesis about KNN being sensitive to scaling was strongly confirmed. Without scaling, KNN achieved:

- Precision: **1.000**
- Recall: **0.021**
- F1-score: **0.041**

After scaling, KNN achieved:

- Precision: **0.956**
- Recall: **0.684**
- F1-score: **0.798**

This large improvement in Recall confirms that feature scaling is extremely important for KNN.

Our hypothesis about Decision Tree overfitting was also confirmed. The unrestricted Decision Tree achieved:

```text
Train F1 = 1.000
Test F1  = 0.713
```

### 2. Which model performed best?

Our initial hypothesis expected Logistic Regression or Decision Tree to perform best. However, based on our experiments, scaled KNN with `k=5` provided the best overall performance.

### 3. Which metric was most informative?

Our initial hypothesis was that Recall would be particularly important, while F1-score would provide a useful balance between Precision and Recall.

The results supported this hypothesis.

Because fraud detection is highly imbalanced, Accuracy can be misleading. A model can achieve approximately 99.8% Accuracy while still missing most fraudulent transactions.

Recall was particularly informative because it measures how many actual fraudulent transactions were detected.

However, using Recall alone could encourage the model to classify too many legitimate transactions as fraud. Therefore, F1-score was useful for selecting a model that balances Recall and Precision.

For this reason, we primarily considered Recall and F1-score, while also monitoring Precision and the confusion matrix.

### 4. How did class imbalance affect the results?

Our hypothesis was confirmed.

The dataset contains approximately:

```text
99.83% legitimate transactions
0.17% fraudulent transactions
```

Therefore, a model predicting almost everything as legitimate could achieve very high Accuracy while being practically useless.

For example, predicting every transaction as Class 0 would result in:

- Accuracy ≈ 99.83%
- Recall = 0

This demonstrates why Accuracy alone was not appropriate for this problem.

### 5. What was the trade-off between False Positives and False Negatives?

Our threshold experiment demonstrated the expected trade-off.

At threshold `0.5`:

```text
False Positives = 3
False Negatives = 30
Recall          = 0.684
Precision       = 0.956
F1              = 0.798
```

At threshold `0.3`:

```text
False Positives = 7
False Negatives = 23
Recall          = 0.758
Precision       = 0.911
F1              = 0.828
```

Lowering the threshold made the model more likely to classify a transaction as fraudulent.

As a result:

```text
False Negatives: 30 → 23  ↓
False Positives:  3 → 7   ↑
Recall:           0.684 → 0.758 ↑
Precision:        0.956 → 0.911 ↓
F1-score:         0.798 → 0.828 ↑
```

Therefore, we accepted a small increase in False Positives in exchange for detecting more fraudulent transactions.

For a fraud detection system, this is a reasonable trade-off because missing an actual fraudulent transaction (False Negative) can be more costly than incorrectly flagging a legitimate transaction (False Positive).