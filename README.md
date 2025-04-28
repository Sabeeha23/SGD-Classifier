# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Necessary Libraries and Load iris Data set

2.Create a DataFrame from the Dataset

3.Add Target Labels to the DataFrame

4.Split Data into Features (X) and Target (y)

5.Split Data into Training and Testing Sets

6.Initialize the SGDClassifier Model

7.Train the Model on Training Data

8.Make Predictions on Test Data

9.Evaluate Accuracy of Predictions

10.Generate and Display Confusion Matrix

11.Generate and Display Classification Report

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Sabeeha Shaik
RegisterNumber:  212223230176
*/
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```
```
iris = load_iris()
```
```
df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
df['target']= iris.target
```
```
print(df.head())
```
```
X = df.drop('target',axis = 1)
y = df['target']
```
```
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
```
```
sgd_clf = SGDClassifier(max_iter = 1000, tol = 1e-3)
```
```
sgd_clf.fit(X_train, y_train)
```
```
y_pred = sgd_clf.predict(X_test)
```
```
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")
```
```
cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)
```
```
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
print("Sabeeha Shaik")
print(212223230176)
```
## Output:
## df.head()
![image](https://github.com/user-attachments/assets/e0d3a109-1ca8-4b37-b5b0-4729955c05ee)

## Accuracy
![image](https://github.com/user-attachments/assets/71bb7db4-4b5c-4053-b4ee-7d56ea58a403)

## Confusion matrix
![image](https://github.com/user-attachments/assets/df1ca5c5-8300-430c-abb8-ce653b4f1f0c)

## Classification report
![image](https://github.com/user-attachments/assets/ff52d7ae-4369-41ea-9c3c-56085bbc8f8f)

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
