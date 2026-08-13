from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from data_prep import prepare_data
from sklearn.metrics import(accuracy_score,
                            recall_score,
                            precision_score,
                            f1_score,
                            confusion_matrix)

X_train,X_test,X_train_scaled,X_test_scaled,y_train,y_test,scaler = prepare_data("data/creditcard.csv")

logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled,y_train)
y_pred_logistic_model = logistic_model.predict(X_test_scaled)

print("Recall :",recall_score(y_test,y_pred_logistic_model))
print("Precision :",precision_score(y_test,y_pred_logistic_model))
print("F1 Score :",f1_score(y_test,y_pred_logistic_model))
cm_logistic_model = confusion_matrix(y_test,y_pred_logistic_model)
print("confusion matrix :\n",cm_logistic_model)
print("Accuracy :",accuracy_score(y_test,y_pred_logistic_model))

print('_'*100)

# k = 5 (default)
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_scaled,y_train)
y_pred_knn_model = knn_model.predict(X_test_scaled)

print("Recall :",recall_score(y_test,y_pred_knn_model))
print("Precision :",precision_score(y_test,y_pred_knn_model))
print("F1 Score :",f1_score(y_test,y_pred_knn_model))
cm_knn_model = confusion_matrix(y_test,y_pred_knn_model)
print("confusion matrix :\n",cm_knn_model)
print("Accuracy :",accuracy_score(y_test,y_pred_knn_model))

print('_'*100)

# unlimit baseline decision tree (possible overfitting)
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train,y_train)
y_pred_decision_tree_model = decision_tree_model.predict(X_test)

print("Recall :",recall_score(y_test,y_pred_decision_tree_model))
print("Precision :",precision_score(y_test,y_pred_decision_tree_model))
print("F1 Score :",f1_score(y_test,y_pred_decision_tree_model))
cm_decision_tree_model = confusion_matrix(y_test,y_pred_decision_tree_model)
print("confusion matrix :\n",cm_decision_tree_model)
print("Accuracy :",accuracy_score(y_test,y_pred_decision_tree_model))