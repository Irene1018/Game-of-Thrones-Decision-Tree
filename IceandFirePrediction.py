#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydotplus

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix


# In[2]:


#load data
df = pd.read_csv("character-deaths.csv")
df.head(3)


# In[3]:


#check missing value 
df.info()


# In[4]:


#fill in "Death Year" missing value with 0
df = df.rename(columns = {'Death Year':'Death'})
df['Death'] = df.Death.fillna(0)
#fill in 1 replace origin number
df.Death[df.Death>0] = 1


# In[5]:


#fill in "Book Intro Chapter" missing value with 0
df = df.rename(columns = {'Book Intro Chapter':'Intro'})
df['Intro'] = df.Intro.fillna(0)
#df.Intro[df.Intro>0] = 1


# In[6]:


#make allegiances turn into dummy features
df1 = pd.get_dummies(df.Allegiances)


# In[7]:


#organize data
df = pd.concat([df, df1], axis = 1)
df = df.drop(['Allegiances', 'Book of Death', 'Death Chapter'], axis = 1)


# In[8]:


df


# In[9]:


#split train and test data
from sklearn.model_selection import train_test_split
x = df.iloc[:, 2:]
y = df.loc[:, 'Death']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)


# In[10]:


#Make an instance of the Model
clf = DecisionTreeClassifier().fit(x_train, y_train)


# In[11]:


#get importance
importance = clf.feature_importances_
#summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))


# In[12]:


#sort feature importance
plt.figure(figsize=(50, 50))
importances = pd.DataFrame({'feature':x_train.columns,'importance':np.round(clf.feature_importances_,3)}) 
importances = importances.sort_values('importance',ascending=False).set_index('feature')  
importances.plot.bar()


# In[13]:


#select feature by threshold
from sklearn.feature_selection import SelectFromModel

model = SelectFromModel(clf, estimator = 'feature_importances_', threshold = 0.05, prefit=True)
feature_idx = model.get_support()
feature_names = x.columns[feature_idx]
x = model.transform(x)
x = pd.DataFrame(x, columns= feature_names)


# In[32]:


#re-split dataset
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)
clf = DecisionTreeClassifier(max_depth = 8).fit(x_train, y_train)


# In[33]:


clf.score(x_test,y_test)


# In[34]:


#decision tree visualization 
fig, ax = plt.subplots(figsize=(25, 25))
plot = tree.plot_tree(clf, ax=ax, fontsize=12)


# In[35]:


y_pred = clf.predict(x_test)
confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)


# In[36]:



y_pred = clf.predict(x_test)

disp = plot_confusion_matrix(clf, x_test, y_test, cmap=plt.cm.Blues)


# In[37]:


tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
Accuracy = (tp+tn)/(tp+fp+fn+tn)
Precision = tp/(tp+fp)
Recall = tp/(tp+fn)

print ("Accuracy = ", Accuracy)
print ("Precision = ", Precision)
print("Recall = ", Recall)

