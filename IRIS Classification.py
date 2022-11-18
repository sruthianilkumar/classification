#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[3]:


data = pd.read_csv('iris.csv')


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


data['Species'].value_counts()


# In[7]:


import seaborn as sns
sns.pairplot(data,hue='Species')
plt.show()


# In[11]:


x = data.drop(['Id', 'Species'], axis=1)
y = data['Species']
print(x.head())


# In[14]:


features=list(data.columns)
features.remove('Id')
features.remove('Species')


# In[15]:


features


# In[16]:


#splitting of data into training and testing

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[17]:


x_train.shape


# In[18]:


x_test.shape


# In[19]:


y_train.shape


# In[20]:


y_test.shape


# In[21]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

# feeding the into the scaler

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[22]:


#x_train


# In[23]:


from sklearn.svm import SVC
from sklearn import metrics


# In[24]:


model = SVC()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


# In[25]:


y_pred


# In[26]:


print("training accuracy :", model.score(x_train, y_train))
print("testing accuracy :", model.score(x_test, y_test))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[27]:


from sklearn.metrics import accuracy_score

print("Accuracy: " ,accuracy_score(y_test, y_pred)*100)


# In[28]:


# Importing classification report
from sklearn.metrics import classification_report

# printing the report
print(classification_report(y_test, y_pred))


# In[32]:


#model = SVC()

new=np.array([[6.9,3.1,4.9,1.5], [5.9,3,4.2,1.5], [6,2.2,5,1.5]])
prediction=model.predict(new)
print("prediction: {}".format(prediction))


# In[ ]:





# In[ ]:




