#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install joblib


# In[15]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib


# In[17]:


df = pd.read_csv("diabetes.csv")
df


# In[19]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
import numpy as np

# Load dataset
X, y = load_iris(return_X_y=True)

# Initialize the Decision Tree Classifier
dtc = DecisionTreeClassifier()

# Perform 5-fold cross-validation and calculate mean accuracy
accuracies = cross_val_score(dtc, X, y, cv=5, scoring='accuracy')

# Print mean accuracy
print(f'Mean Accuracy: {accuracies.mean():.4f}')


# In[21]:


# Preprocess
X = df.drop('class', axis=1)
y = df['class']


# In[23]:


# standardize X data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled


# In[31]:


# K- flod cross-validation
model = DecisionTreeClassifier(random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_scaled, y, cv=kf)

print("Cross-validation scores:", scores)
print("Mean accuarcy:", scores.mean())


# In[33]:


# Train final model
model.fit(X_scaled, y)

# save model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




