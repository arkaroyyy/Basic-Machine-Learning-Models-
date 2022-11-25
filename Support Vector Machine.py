#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing


# In[8]:


test_data=pd.read_csv()
train_data=pd.read_csv()


# In[9]:


dftest=pd.DataFrame(test_data)
dftrain=pd.DataFrame(train_data)


# In[10]:


#splitting dataset into attributes and lables
x=dftrain.iloc[:, :-1].values
y=dftrain.iloc[:, 4].values
x1=dftest.iloc[:, :-1].values
y1=dftest.iloc[:, 4].values


# In[39]:


print("X Values:\n",x)
print("Y Values:\n",y)


# In[17]:


#scaling the features for uniform evaluation
from sklearn.preprocessing import StandardScaler

#creating object
scaler = StandardScaler()

#fit data
scaler.fit(x)

#transform data
x_train = scaler.transform(x)
x_test = scaler.transform(x1)


# In[30]:


#Normalising
from sklearn.preprocessing import normalize

#creating object
traindata_normalised = normalize(x_train)
testdata_normalised = normalize(x_test)


# In[31]:


##applying the Support Vector Machine algorithm to the dataset
from sklearn.svm import SVC


# In[32]:


#using linear kernel as its faster to train
K=SVC(kernel="linear")


# In[33]:


#training the dataset
K.fit(traindata_normalised,y)


# In[40]:


#predicting the dataset
y_pred=K.predict(testdata_normalised)


# In[36]:


#evaluating predictions and accuracy using confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y1,y_pred)


# In[41]:


accuracy_score(y1,y_pred)


# In[57]:


#evaluating the algorithm, using confusion matrix and classification report
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
print(confusion_matrix(y1, y_pred))
print(classification_report(y1, y_pred))


# In[85]:


plt.scatter(x1[:, 0], x1[:, 2], c = "green")
plt.colorbar(ticks=[0, 1, 2])


# In[87]:





# In[54]:


import seaborn as sns
name=[0,1]
fig, ax = plt.subplots()
marks = np.arange(len(name))
plt.xticks(marks, name)
plt.yticks(marks, name)
sns.heatmap(pd.DataFrame(confusion_matrix(y1,y_pred)), annot=True, cmap="Greens",
   fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

