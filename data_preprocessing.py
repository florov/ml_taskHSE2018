# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 20:25:44 2018

@author: pcborg
"""
# 1 task - data preprocessing

# importting the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as skl

# importing the dataset

dataset = pd.read_csv('who_suicide_statistics.csv')

# create matrix of features X and dependent var = y

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, :5].values

# handle missing data via 'mean'

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis= 0)
imputer = imputer.fit(X[:, 4:6 ])
X[:, 4:6] = imputer.transform(X[:, 4:6])

# handle categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# splitting the dataset into training set and test set

from sklearn.cross_validation import train_test_split
X_train, X_test, t_train, y_test = train_test_split(X, y, test_size = 0.25)

# balance scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

'Dear Pavel! 
'I tried to use these steps on my first data set, but it did not even import accurate,'
' so i did switch to another data (with more simple structure),'
' but it is still dont work perfectly, otherwise i check my code on very simple data and it did work!'
' So i do my best to repair my code and data and to do next task till Saturday and i hope to ask some questions about my work if it will be time for this'
