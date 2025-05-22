# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the Program.
2.Import the necessary packages.
3.Read the given csv file and display the few contents of the data.
4.Assign the features for x and y respectively.
5.Split the x and y sets into train and test sets.
6.Convert the Alphabetical data to numeric using CountVectorizer.
7.Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
8.Find the accuracy of the model.
9.End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: HARIPRASHAAD RA
RegisterNumber:  212223040060
*/
```

```
import pandas as pd
data = pd.read_csv("spam.csv", encoding="Windows-1252")
data.info()

x = data['v2'].values
y = data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
x_train

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
acc

```

## Output:
![image](https://github.com/user-attachments/assets/bd6146ba-2b7f-424b-a629-7e3d3be1fd8d)
![image](https://github.com/user-attachments/assets/005d7da1-a3cf-4a83-a2ba-264c1052337f)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
