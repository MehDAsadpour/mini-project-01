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

from data_prep import prepare_data,get_scaler

def evaluate_model(model, X_test, y_test,model_name):
    y_pred = model.predict(X_test)

    print(f"\n==={model_name}===\n")
    print("Recall :", recall_score(y_test, y_pred))
    print("Precision :", precision_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix :\n", cm)

    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("_" * 100)


def evaluate_train_model(model, X_train, y_train):
    y_pred_train = model.predict(X_train)

    print("Train Recall :", recall_score(y_train, y_pred_train))
    print("Train Precision :", precision_score(y_train, y_pred_train))
    print("Train F1 Score :", f1_score(y_train, y_pred_train))


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

#==============================================================
# Experiment 2: KNN Hyperparameter Analysis
#==============================================================

for k in [1, 5, 20]:
    knn_pipeline.set_params(model__n_neighbors=k)

    knn_pipeline.fit(X_train, y_train)

    evaluate_model(knn_pipeline, X_test, y_test,f"KNN Hyperparameter Analysis : k = {k}")