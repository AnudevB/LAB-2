#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np #importing library numpy
import pandas as pd  #importing library pandas

col_names=["sepal_length","sepal_width","petal_length","petal_width","type"]#creating a list 
data=pd.read_csv("iris_dataset.csv",skiprows=1,header=None,names=col_names)
#to read a file using  the python library pandas,if header =None is not given then the first value wil get into header
#we are giving 'name=col_names' for making the first header as 'col_names' for making the title in fullform as it was different in file
data.head(20) #just for displaying the first 20 values


# In[43]:


X=data.iloc[:,:-1].values #To select all columns except the last column (-1 means last column)
Y1=data.iloc[:,-1].values # taking the last column as output
Y=Y1.reshape(-1,1) # to reshape Y1 from 1D array to 2D array with a single column(1 means there should be only one column)

from  sklearn.model_selection import train_test_split #importing ' train_test_split' to split the data for training and testing
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=41)#Here we are splitting the data into training and testing ,'test_size=0.3' means 30% of the data is used for testing and the rest 70% for training
#X_train,Y_train contain  data for the training set
#X_test,Y_test contain  data for the testing set

from sklearn import tree #importing 'tree' for decision tree modeling.
classifier=tree.DecisionTreeClassifier(min_samples_split=3,max_depth=3,criterion="entropy")#Restricting the decision tree upto three layers and using the criterion 'entropy'
classifier.fit(X_train,Y_train)# for training the model using the data X_train and Y_train.
x=classifier.score(X_test,Y_test)#Gives the accuracy of the decision tree model using the datas X_test and Y_test(value between 0 and 1)
print("Accuracy=",x)
tree.plot_tree(classifier)# to plot the tree


# In[ ]:




