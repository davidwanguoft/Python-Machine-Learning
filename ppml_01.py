#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 16:16:23 2018
@author: dwang
"""

import pandas as pd 
import quandl, math
quandl.ApiConfig.api_key = 'REDACTED'
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

import datetime
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import pickle

df = quandl.get("WIKI/GOOGL")


# pruning
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

# dimension creation

df['HL_pct']=(df['Adj. High']-df['Adj. Low'])/df['Adj. Low'] * 100.0
df['PCT_change'] =(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] * 100.0

forecast_col = 'Adj. Close'
df.fillna(value=-999999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))

# forecast out 1% of the entire length of the dataset
df['label'] = df[forecast_col].shift(-forecast_out)


X = np.array(df.drop(['label'], 1)) #features
X = preprocessing.scale(X) # normalize -1 to +1
X_lately = X[-forecast_out:] # most recent features (predict against)
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label']) #labels

# 20% test group size
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# 2. COMMENT THIS PART OUT AFTER PICKLED; SERIALIZED CLASSIFIER SAVED
#clf = LinearRegression(n_jobs=1) # use all available threads
#clf.fit(X_train, y_train)
#confidence = clf.score(X_test, y_test)
#print("Confidence score: ",confidence)

# 1. to pickle the classifier above; comment out once pickle file is generated
#with open('linearregression.pickle','wb') as f:
#    pickle.dump(clf, f)

# 2. load classifier in via pickle; classifier above is no longer needed since it's been serialized
pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan # will populate later
print(forecast_set, confidence, forecast_out)

# grab last day in df, then start assigning forecasts to new days
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]


df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


'''
# testconfidence
for k in ['linear','poly','rbf','sigmoid']:
    clf = svm.SVR(kernel=k)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print (k, confidence)
#clf = svm.SVR() # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
''' 
