#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[1]:


test_data=pd.read_csv()
train_data=pd.read_csv()


# In[4]:


dftest=pd.DataFrame(test_data)
dftrain=pd.DataFrame(train_data)


# In[5]:


#splitting dataset into attributes and lables
x=dftrain.iloc[:, :-1].values
y=dftrain.iloc[:, 4].values
x1=dftest.iloc[:, :-1].values
y1=dftest.iloc[:, 4].values


# In[6]:


print("X Values:\n",x)


# In[7]:


print("Y Values:\n",y)


# In[8]:


#scaling the features for uniform evaluation
from sklearn.preprocessing import StandardScaler

#creating object
scaler = StandardScaler()

#fit data
scaler.fit(x)

#transform data
x_train = scaler.transform(x)
x_test = scaler.transform(x1)


# In[9]:


#Normalising
from sklearn.preprocessing import normalize

#creating object
traindata_normalised = normalize(x_train)
testdata_normalised = normalize(x_test)


# In[27]:


#applying the KNN algorithm to the dataset
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train, y)


# In[17]:


#predicting the test data
y_pred = classifier.predict(x_test)
y_pred


# In[15]:


#checking accuracy
from sklearn.metrics import accuracy_score
accuracy =accuracy_score(y1,y_pred)
print("Accuracy for normalised:",accuracy)


# In[14]:


#evaluating the knn algorithm, using confusion matrix and classification report
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y1, y_pred))
print(classification_report(y1, y_pred))


# In[43]:


from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#00AAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00','#00AAFF'])
clf = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')
clf.fit(x, y)


# In[ ]:





# In[23]:


import seaborn as sns
name=[0,1]
fig, ax = plt.subplots()
marks = np.arange(len(name))
plt.xticks(marks, name)
plt.yticks(marks, name)
sns.heatmap(pd.DataFrame(confusion_matrix(y1,y_pred)), annot=True, cmap="BuPu",
   fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion Matrix using knn', y=1.1)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

