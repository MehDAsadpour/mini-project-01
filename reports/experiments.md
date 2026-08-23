# Model Experiments

This document contains the experiments performed during the development of the Credit Card Fraud Detection model.

The main goals of these experiments were:

- Compare the baseline performance of different models.
- Evaluate the effect of feature scaling on KNN.
- Analyze the effect of KNN hyperparameters.
- Analyze Decision Tree complexity and overfitting.
- Study the effect of classification thresholds.
- Validate the final model using cross-validation.
- Select a suitable final model and threshold.

The main evaluation metrics used throughout the experiments are:

- Precision
- Recall
- F1-score
- Accuracy
- Confusion Matrix

Because the dataset is highly imbalanced, Accuracy was not used as the primary metric for model selection.

---

# 1. Baseline Model Comparison

The first experiment compares the three baseline models:

- Logistic Regression
- KNN with `k=5` and feature scaling
- Decision Tree with `max_depth=None`

The purpose of this experiment was to establish a baseline before performing further experiments.

## 1.1 Test Set Results

| Model | Precision | Recall | F1-score | Accuracy |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.846 | 0.579 | 0.688 | 0.9991 |
| KNN (k=5) | **0.956** | 0.684 | **0.798** | **0.9994** |
| Decision Tree (max_depth=None) | 0.720 | **0.705** | 0.713 | 0.9990 |

### Logistic Regression

Logistic Regression achieved:

- Precision: `0.846`
- Recall: `0.579`
- F1-score: `0.688`

The model detected approximately 58% of fraudulent transactions. Although its Precision was relatively high, its Recall was lower than both KNN and Decision Tree.

### KNN

KNN with `k=5` and feature scaling achieved:

- Precision: `0.956`
- Recall: `0.684`
- F1-score: `0.798`

KNN provided the highest F1-score among the three baseline models and also achieved the highest Precision.

### Decision Tree

The unrestricted Decision Tree achieved:

- Precision: `0.720`
- Recall: `0.705`
- F1-score: `0.713`

Its Recall was slightly higher than KNN, but its Precision was significantly lower. This means that the model detected a similar number of fraudulent transactions but generated more False Positives.

### Baseline Conclusion

Based on the baseline test results, KNN provided the best overall balance between Precision and Recall.

However, the baseline results alone were not sufficient for selecting the final model. Further experiments were required to investigate scaling, hyperparameters, overfitting, and classification thresholds.

---

# 2. Cross-Validation of Baseline Models

To evaluate whether the baseline results were consistent across different subsets of the training data, 5-Fold Stratified Cross-Validation was performed.

`StratifiedKFold` was used to preserve the class distribution in each fold.

## 2.1 Cross-Validation Results

| Model | Mean Precision | Mean Recall | Mean F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.863 | 0.606 | 0.710 |
| KNN (k=5) | **0.916** | **0.765** | **0.832** |
| Decision Tree | 0.758 | 0.743 | 0.750 |

KNN achieved the best mean Precision, Recall, and F1-score across the five folds.

This provides stronger evidence that the good performance of KNN was not caused only by a particular train/test split.

The difference between KNN and the other models was especially noticeable in Mean F1-score:

- Logistic Regression: `0.710`
- Decision Tree: `0.750`
- KNN: `0.832`

Therefore, KNN was selected for further investigation.

---

# 3. Experiment 1: Effect of Feature Scaling on KNN

The first mandatory experiment investigated whether feature scaling affects KNN performance.

KNN is a distance-based algorithm. Therefore, features with larger numerical scales can dominate the distance calculation.

Two versions of KNN were compared:

1. KNN without scaling
2. KNN with StandardScaler

## 3.1 Results

| Model | Scaling | Precision | Recall | F1-score | Accuracy |
|---|---|---:|---:|---:|---:|
| KNN | Without Scaling | **1.000** | 0.021 | 0.041 | 0.9984 |
| KNN | With Scaling | 0.956 | **0.684** | **0.798** | **0.9994** |

The difference is significant.

Without scaling, KNN achieved a Precision of `1.000`, but this result is misleading because the model detected only:

```text
Recall = 0.021
```
This means that the model detected only about 2% of the actual fraudulent transactions.

After applying `StandardScaler`:

Recall = 0.684
F1-score = 0.798

The improvement in Recall demonstrates that scaling is extremely important for KNN in this dataset.

## 3.2 Conclusion

The hypothesis that KNN would be sensitive to feature scaling was strongly confirmed.

The reason is that KNN determines neighboring samples using distance calculations. If features have different scales, large-scale features can dominate the distance calculation and prevent the algorithm from properly using the information contained in other features.

Therefore, the final KNN model uses `StandardScaler` as part of the preprocessing process.

# 4. Experiment 2: KNN Hyperparameter Analysis

The second mandatory experiment investigated the effect of the number of neighbors (`k`) on KNN performance.

The following values were tested:

`k = 1`, `k = 5`, `k = 20`

## 4.1 Test Set Results

| K | Precision | Recall | F1-score | Accuracy |
|---|---:|---:|---:|---:|
| 1 | 0.829 | **0.716** | 0.768 | 0.9993 |
| 5 | **0.956** | 0.684 | **0.798** | **0.9994** |
| 20 | 0.915 | 0.684 | 0.783 | 0.9994 |

### k = 1

With `k=1`, the model achieved the highest Recall:

`Recall = 0.716`

However, Precision decreased significantly:

`Precision = 0.829`

This suggests that using only one nearest neighbor makes the model more sensitive to individual samples.

### k = 5

With `k=5`, the model achieved:

`Precision = 0.956`

`Recall = 0.684`

`F1-score = 0.798`

This produced the highest F1-score among the tested values.

### k = 20

Increasing `k` to 20 produced:

`Precision = 0.915`

`Recall = 0.684`

`F1-score = 0.783`

The F1-score was slightly lower than the result obtained with `k=5`.

## 4.2 Conclusion

Based on the balance between Precision and Recall, `k=5` provided the best overall performance.

Therefore:

`Final KNN hyperparameter: k = 5`

# 5. Experiment 3: Decision Tree Complexity and Overfitting

The next experiment investigated whether an unrestricted Decision Tree would overfit.

The following values of `max_depth` were tested:

`max_depth = 2`, `max_depth = 5`, `max_depth = 10`, `max_depth = None`

Both training and test performance were recorded.

## 5.1 Test Set Results

| max_depth | Precision | Recall | F1-score | Accuracy |
|---|---:|---:|---:|---:|
| 2 | 0.835 | 0.695 | 0.759 | 0.9993 |
| 5 | **0.882** | **0.705** | **0.784** | **0.9993** |
| 10 | 0.892 | 0.695 | 0.781 | 0.9993 |
| None | 0.720 | 0.705 | 0.713 | 0.9990 |

The best test F1-score was achieved with:

`max_depth = 5`

`F1-score = 0.784`

## 5.2 Training Set Results

| max_depth | Train Precision | Train Recall | Train F1 | Train Accuracy |
|---|---:|---:|---:|---:|
| 2 | 0.822 | 0.746 | 0.782 | 0.9993 |
| 5 | 0.942 | 0.817 | 0.875 | 0.9996 |
| 10 | 0.991 | 0.873 | 0.928 | 0.9998 |
| None | **1.000** | **1.000** | **1.000** | **1.0000** |

As `max_depth` increased, the training performance continuously improved.

However, the test performance did not improve in the same way.

For example:

`max_depth = 10`

`Train F1 = 0.928`

`Test F1 = 0.781`

For the unrestricted tree:

`max_depth = None`

`Train F1 = 1.000`

`Test F1 = 0.713`

The unrestricted tree perfectly fitted the training data but performed considerably worse on unseen data.

## 5.3 Conclusion

The results confirm the initial hypothesis that an unrestricted Decision Tree is likely to overfit.

Increasing tree complexity allowed the model to fit the training data more closely, but this did not result in better generalization.

The large difference between training and test performance for `max_depth=None` is strong evidence of overfitting.

# 6. Experiment 4: Classification Threshold Analysis

After selecting KNN with `k=5`, different classification thresholds were investigated.

The default classification threshold is:

`threshold = 0.5`

However, in a fraud detection problem, the threshold can be adjusted to change the balance between False Positives and False Negatives.

---

# 7. Logistic Regression Threshold Analysis

The following thresholds were tested:

`0.3`, `0.5`, `0.7`

| Threshold | Precision | Recall | F1-score |
|---|---:|---:|---:|
| 0.3 | 0.836 | **0.642** | **0.726** |
| 0.5 | **0.846** | 0.579 | 0.688 |
| 0.7 | 0.845 | 0.516 | 0.641 |

Lowering the threshold from `0.5` to `0.3` increased Recall:

`0.579 → 0.642`

while Precision decreased slightly:

`0.846 → 0.836`

This demonstrates the expected Precision-Recall trade-off.

# 8. KNN Threshold Analysis

For KNN with `k=5`, the following thresholds were tested:

`0.3`, `0.4`, `0.5`, `0.6`, `0.7`

## 8.1 Results

| Threshold | Precision | Recall | F1-score |
|---|---:|---:|---:|
| 0.3 | 0.883 | **0.716** | **0.791** |
| 0.4 | 0.893 | 0.705 | 0.788 |
| 0.5 | 0.904 | 0.695 | 0.786 |
| 0.6 | 0.926 | 0.663 | 0.773 |
| 0.7 | **0.952** | 0.621 | 0.752 |

The results show a clear trade-off.

As the threshold increases:

- Precision increases.
- Recall decreases.

A lower threshold makes the model more willing to classify a transaction as fraudulent, increasing fraud detection but also increasing False Positives.

The highest F1-score among the tested thresholds was achieved at:

`threshold = 0.3`

`F1-score = 0.791`

Therefore, threshold `0.3` was selected for further cross-validation.

---

# 9. Cross-Validation of KNN Thresholds

The threshold experiment was also evaluated using 5-Fold Stratified Cross-Validation.

Two thresholds were compared:

`threshold = 0.3`

`threshold = 0.5`

## 9.1 Results

| Threshold | Mean Precision | Mean Recall | Mean F1 |
|---|---:|---:|---:|
| 0.3 | 0.822 | **0.791** | **0.805** |
| 0.5 | **0.845** | 0.767 | 0.802 |

Threshold `0.3` achieved:

- Higher Mean Recall
- Slightly higher Mean F1-score

Threshold `0.5` achieved:

- Higher Mean Precision

The difference in Mean F1-score was small:

`Threshold 0.3 → 0.805`

`Threshold 0.5 → 0.802`

However, threshold `0.3` provided better Recall:

`0.791 vs 0.767`

Since fraud detection places significant importance on detecting fraudulent transactions, threshold `0.3` provides a more suitable trade-off.

# 10. False Positive vs False Negative Trade-off

The threshold experiment clearly demonstrated the trade-off between False Positives and False Negatives.

For KNN at threshold `0.5`:

False Positives = 7  
False Negatives = 29  
Recall = 0.695  
Precision = 0.904  
F1-score = 0.786

At threshold `0.3`:

False Positives = 9  
False Negatives = 27  
Recall = 0.716  
Precision = 0.883  
F1-score = 0.791

Lowering the threshold resulted in:

False Negatives: 29 → 27  
False Positives: 7 → 9  
Recall: 0.695 → 0.716  
Precision: 0.904 → 0.883  
F1-score: 0.786 → 0.791

Therefore, we accepted a small increase in False Positives in exchange for reducing False Negatives and detecting more fraudulent transactions.

For a fraud detection problem, this is a reasonable trade-off because missing an actual fraudulent transaction can be more costly than incorrectly flagging a legitimate transaction.

# 11. Effect of Class Imbalance

The dataset is extremely imbalanced:

Legitimate transactions ≈ 99.83%  
Fraudulent transactions ≈ 0.17%

This imbalance has a major effect on model evaluation.

A model that predicts every transaction as legitimate would achieve approximately:

Accuracy ≈ 99.83%  
Recall = 0

Such a model would appear highly accurate but would be completely ineffective for fraud detection.

Therefore, Accuracy was not used as the main criterion for model selection.

Instead, the experiments focused primarily on:

- Precision
- Recall
- F1-score
- Confusion Matrix

Recall was particularly important because False Negatives represent fraudulent transactions that the system failed to detect.

F1-score was also important because it provides a balance between Precision and Recall.

# 12. Overall Experimental Findings

The experiments produced several important findings.

## Finding 1: KNN requires feature scaling

The comparison between scaled and unscaled KNN strongly confirmed this.

Without scaling:

Recall = 0.021  
F1-score = 0.041

With scaling:

Recall = 0.684  
F1-score = 0.798

Therefore, `StandardScaler` is essential for the final KNN pipeline.

---

## Finding 2: KNN with k=5 provided the best balance

Among the tested K values:

`k = 1 → F1 = 0.768`  
`k = 5 → F1 = 0.798`  
`k = 20 → F1 = 0.783`

Therefore, `k=5` provided the best test-set F1-score.

---

## Finding 3: Decision Tree overfitting was confirmed

The unrestricted Decision Tree achieved:

`Train F1 = 1.000`  
`Test F1 = 0.713`

This large gap indicates strong overfitting.

Limiting `max_depth` improved generalization, with `max_depth=5` achieving the highest test F1-score among the tested tree depths.

---

## Finding 4: Threshold adjustment affects the Precision-Recall trade-off

For KNN, increasing the threshold resulted in higher Precision but lower Recall.

For example:

`Threshold = 0.3`  
`Precision = 0.883`  
`Recall = 0.716`

`Threshold = 0.7`  
`Precision = 0.952`  
`Recall = 0.621`

Therefore, threshold selection should depend on the requirements of the fraud detection system.

---

## Finding 5: Cross-validation supports KNN

The baseline cross-validation results were:

| Model | Mean Precision | Mean Recall | Mean F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.863 | 0.606 | 0.710 |
| KNN | **0.916** | **0.765** | **0.832** |
| Decision Tree | 0.758 | 0.743 | 0.750 |

KNN achieved the best average performance across all three main metrics.

This makes the KNN results more reliable than relying only on a single train/test split.

# 13. Final Experimental Conclusion

Based on all experiments, the final model configuration was selected as:

Model: KNN  
n_neighbors: 5  
Preprocessing: StandardScaler  
Threshold: 0.3

The main reasons for this selection were:

1. KNN achieved the highest baseline F1-score.
2. KNN achieved the strongest cross-validation performance.
3. Feature scaling produced a very large improvement in KNN Recall.
4. `k=5` provided the best balance among the tested K values.
5. Threshold `0.3` provided higher Recall and a slightly higher F1-score than threshold `0.5` during cross-validation.
6. The lower threshold reduces False Negatives, which is desirable for fraud detection.
7. Decision Tree showed clear signs of overfitting at higher complexity.
8. Accuracy was not considered sufficient because of the severe class imbalance.

The final model therefore prioritizes a balance between detecting fraudulent transactions and avoiding excessive False Positives.