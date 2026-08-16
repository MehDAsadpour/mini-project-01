from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix
)
import joblib
from data_prep import prepare_data,get_scaler

def evaluate_model(model, X_test, y_test,model_name):
    y_pred = model.predict(X_test)

    print(f"\n*** {model_name} ***\n")
    print("Recall :", recall_score(y_test, y_pred))
    print("Precision :", precision_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix :\n", cm)

    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("_" * 100)


def evaluate_train_model(model, X_train, y_train,model_name):
    y_pred_train = model.predict(X_train)

    print(f"\n** {model_name} **\n")
    print("Train Recall :", recall_score(y_train, y_pred_train))
    print("Train Precision :", precision_score(y_train, y_pred_train))
    print("Train F1 Score :", f1_score(y_train, y_pred_train))
    print("Train Accuracy :", accuracy_score(y_train, y_pred_train))
    print("_" * 100)


def cross_validate_model(model, X, y, cv,model_name):
    results = cross_validate(
        model,
        X,y,
        cv=cv,
        scoring= ["precision","recall","f1"]
    )
    mean_precision = results["test_precision"].mean()
    mean_recall = results["test_recall"].mean()
    mean_f1 = results["test_f1"].mean()

    print(f"\n==={model_name}===\n")
    print("Mean Precision :", mean_precision)
    print("Mean Recall :", mean_recall)
    print("Mean F1 :", mean_f1)
    print("_" * 100)


def evaluate_threshold(model, X_test, y_test, threshold):
    y_proba = model.predict_proba(X_test)[:, 1]

    y_pred = (y_proba >= threshold).astype(int)

    print(f"\n=== Threshold = {threshold} ===")

    print("Recall :", recall_score(y_test, y_pred))
    print("Precision :", precision_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))

    print("Confusion Matrix :\n", confusion_matrix(y_test, y_pred))
    print("_" * 100)


def cross_validate_threshold(model, X, y, cv, threshold, model_name):
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for train_idx, val_idx in cv.split(X, y):

        X_train_cv = X.iloc[train_idx]
        X_val_cv = X.iloc[val_idx]

        y_train_cv = y.iloc[train_idx]
        y_val_cv = y.iloc[val_idx]

        model.fit(X_train_cv, y_train_cv)

        y_proba = model.predict_proba(X_val_cv)[:, 1]

        y_pred = (y_proba >= threshold).astype(int)

        precision_scores.append(
            precision_score(y_val_cv, y_pred)
        )

        recall_scores.append(
            recall_score(y_val_cv, y_pred)
        )

        f1_scores.append(
            f1_score(y_val_cv, y_pred)
        )

    print(f"\n=== {model_name} | Threshold = {threshold} ===")

    print("Mean Precision :", sum(precision_scores) / len(precision_scores))
    print("Mean Recall :", sum(recall_scores) / len(recall_scores))
    print("Mean F1 :", sum(f1_scores) / len(f1_scores))

    print("_" * 100)


X_train,X_test,y_train,y_test = prepare_data(
    "data/creditcard.csv"
    )


#==============================================================
# BaseLine Models
#==============================================================

# Logistic Regression
logistic_pipeline = Pipeline([
    ("scaler", get_scaler()),
    ("model", LogisticRegression(random_state=42))
])
logistic_pipeline.fit(X_train, y_train)
evaluate_model(
    logistic_pipeline,
    X_test,
    y_test,
    "Logistic Regression - Baseline"
)

# k = 5 (default) with Scaling
knn_pipeline = Pipeline([
    ("scaler", get_scaler()),
    ("model", KNeighborsClassifier())
])
knn_pipeline.fit(X_train, y_train)
evaluate_model(
    knn_pipeline,
    X_test,
    y_test,
    "KNN - Baseline (k=5, with Scaling)"
)

# Unlimited-depth baseline Decision Tree (possible overfitting)
decision_tree_model = DecisionTreeClassifier(
    random_state=42
)
decision_tree_model.fit(X_train, y_train)
evaluate_model(
    decision_tree_model,
    X_test,
    y_test,
    "Decision Tree - Baseline (max_depth=None)"
)


#==============================================================
# Cross Validation BaseLine Models
#==============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cross_validate_model(
    logistic_pipeline,
    X_train,y_train,
    cv,
    "Cross Validation - Logistic Regression"
)

cross_validate_model(
    knn_pipeline,
    X_train,
    y_train,
    cv,
    "Cross Validation - KNN (k=5)"
)

cross_validate_model(
    decision_tree_model,
    X_train,
    y_train,
    cv,
    "Cross Validation - Decision Tree"
)


#==============================================================
# Experiment 1: Effect of Scaling
#==============================================================

# KNN without Scaling
knn_no_scale = KNeighborsClassifier()
knn_no_scale.fit(X_train,y_train)
evaluate_model(
    knn_no_scale,
    X_test,
    y_test,
    "Experiment 1 - KNN without Scaling"
)

# #==============================================================
# # Experiment 2: KNN Hyperparameter Analysis
# #==============================================================

# # KNN
for k in [1, 5, 20]:
    knn_pipeline.set_params(model__n_neighbors=k)

    knn_pipeline.fit(X_train, y_train)

    evaluate_model(knn_pipeline, X_test, y_test,f"KNN Hyperparameter Analysis : k = {k}")

#Decision Tree
decision_tree_model = DecisionTreeClassifier(
    random_state=42
)

for depth in [2, 5, 10, None]:
    decision_tree_model.set_params(
        max_depth=depth
    )

    decision_tree_model.fit(
        X_train,
        y_train
    )

    evaluate_model(
        decision_tree_model,
        X_test,
        y_test,
        f"Decision Tree - test - max_depth={depth}"
    )

    evaluate_train_model(
        decision_tree_model,
        X_train,
        y_train,
        f"Decision Tree - Train - max_depth={depth}"
    )

# Logistic Regression
print("\n<<<< Logistic Regression Threshold Analysis >>>>\n")
for threshold in [0.3, 0.5, 0.7]:
    evaluate_threshold(
        logistic_pipeline,
        X_test,
        y_test,
        threshold
)

# Finilizing
print("\n<<<< KNN Threshold Analysis >>>>\n")
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_proba = knn_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print(f"\n=== Threshold = {threshold} ===")

    print("Recall :", recall_score(y_test, y_pred))
    print("Precision :", precision_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))
    print("Confusion Matrix :\n", confusion_matrix(y_test, y_pred))
    print("_"*100)

print("\n<<<< CV for KNN with threshold {0.3} >>>>\n")
cross_validate_threshold(
    knn_pipeline,
    X_train,
    y_train,
    cv,
    threshold=0.3,
    model_name="KNN - k=5"
)
print("\n<<<< CV for KNN with threshold {0.5} >>>>\n")
cross_validate_threshold(
    knn_pipeline,
    X_train,
    y_train,
    cv,
    threshold=0.5,
    model_name="KNN - k=5"
)

# ==============================================================
# Final Model
# ==============================================================

final_knn_pipeline = Pipeline([
    ("scaler", get_scaler()),
    ("model", KNeighborsClassifier(n_neighbors=5))
])

final_knn_pipeline.fit(X_train, y_train)

joblib.dump(
    final_knn_pipeline,
    "models/model.pkl"
)