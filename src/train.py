from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from data_prep import prepare_data
from sklearn.metrics import(accuracy_score,
                            recall_score,
                            precision_score,
                            f1_score,
                            confusion_matrix)

def evaluate_model(model,X_test,y_test):
    y_pred = model.predict(X_test)
    print("Recall :",recall_score(y_test,y_pred))
    print("Precision :",precision_score(y_test,y_pred))
    print("F1 Score :",f1_score(y_test,y_pred))
    cm = confusion_matrix(y_test,y_pred)
    print("confusion matrix :\n",cm)
    print("Accuracy :",accuracy_score(y_test,y_pred))
    print('_'*100)

X_train,X_test,X_train_scaled,X_test_scaled,y_train,y_test,scaler = prepare_data(
    "data/creditcard.csv"
    )

logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train_scaled,y_train)
evaluate_model(logistic_model,X_test_scaled,y_test)

# k = 5 (default)
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_scaled,y_train)
evaluate_model(knn_model,X_test_scaled,y_test)

# Unlimited-depth baseline Decision Tree (possible overfitting)
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train,y_train)
evaluate_model(decision_tree_model,X_test,y_test)