#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 20:58:14 2018

@author: dwang
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing, neighbors, cross_validation

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-999999, inplace=True)
df.drop(['id'], 1, inplace=True)

print(df.columns.values.tolist())



X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])



X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)
clf = neighbors.KNeighborsClassifier()

clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print('Accuracy score is:', accuracy*100,'%')

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures),-1)
prediction = clf.predict(example_measures)
print('Prediction score is:', prediction)