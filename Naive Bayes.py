#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


test_data=pd.read_csv()
train_data=pd.read_csv()


# In[4]:


dftest=pd.DataFrame(test_data)
dftrain=pd.DataFrame(train_data)


# In[5]:


#splitting dataset into attributes and lables
x=dftrain.iloc[:, :-1].values
y=dftrain.iloc[:, 4].values


# In[6]:


print("X Values:\n",x)
print("Y Values:\n",y)


# In[7]:


#to avoid overfittng, spliting the training and test datatset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.30)


# In[8]:


#scaling the features for uniform evaluation
from sklearn.preprocessing import StandardScaler

#creating object
scaler = StandardScaler()

#fit data
scaler.fit(x_train)

#transform data
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[12]:


##applying the Naive Bayes algorithm to the dataset
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)


# In[13]:


#predicting the labels of the data values on the basis of the trained model
y_pred = classifier.predict(x_test)
print(y_pred)


# In[14]:


y_compare = np.vstack((y_test,y_pred)).T
y_compare[:20,:]


# In[17]:


#evaluating the algorithm, using confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[21]:


#evaluating predictions and accuracy
t = cm.shape
correct = 0
wrong = 0

for row in range(t[0]):
    for x in range(t[1]):
        if row == x:
            correct +=cm[row,x]
        else:
            wrong += cm[row,x]
print('Correct predictions: ', correct)
print('Wrong predictions', wrong)
print ('\nThe accuracy using Naive Bayes Classification is: ', correct/(cm.sum()))    

