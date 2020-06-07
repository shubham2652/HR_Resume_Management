# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 07:00:57 2020

@author: Shubham Shah
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:20:22 2020

@author: Shubham Shah
"""

import pandas as pd
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:,2] = labelencoder_X.fit_transform(X[:,2])
X[:,3] = labelencoder_X.fit_transform(X[:,3])
"""transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        
         OneHotEncoder(), 
         [2,3]              
         )
    ],
    remainder='passthrough' 
)
X = transformer.fit_transform(X.tolist())
X = X.astype('float64')"""
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train[:,[0,1,4]] = sc_X.fit_transform(X_train[:,[0,1,4]])
X_test[:,[0,1,4]] = sc_X.transform(X_test[:,[0,1,4]])
def featurescale(skill,experiance,testScore):
    a = [skill,experiance,testScore]
    featured_value=(sc_X.transform([a]))
    return featured_value
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
from sklearn.externals import joblib
joblib.dump(classifier,'model2.pkl')
