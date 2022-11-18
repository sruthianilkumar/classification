

#Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#Read dataset

from sklearn.datasets import load_iris
data = load_iris()

#print dataset
data.head()

data.info()

data['Species'].value_counts()


import seaborn as sns
sns.pairplot(data,hue='Species')
plt.show()


x = data.drop(['Id', 'Species'], axis=1)
y = data['Species']
print(x.head())


features=list(data.columns)
features.remove('Id')
features.remove('Species')

features


#splitting of data into training and testing

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)



x_train.shape

x_test.shape

y_train.shape

y_test.shape


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

# feeding the into the scaler

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


#SVM

from sklearn.svm import SVC
from sklearn import metrics

model = SVC()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

y_pred

print("training accuracy :", model.score(x_train, y_train))
print("testing accuracy :", model.score(x_test, y_test))


#CONFUSION MATRIX

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

#ACCURACY

from sklearn.metrics import accuracy_score

print("Accuracy: " ,accuracy_score(y_test, y_pred)*100)


# Importing classification report
from sklearn.metrics import classification_report

# printing the report
print(classification_report(y_test, y_pred))


#model = SVC()

new=np.array([[6.9,3.1,4.9,1.5], [5.9,3,4.2,1.5], [6,2.2,5,1.5]])
prediction=model.predict(new)
print("prediction: {}".format(prediction))





