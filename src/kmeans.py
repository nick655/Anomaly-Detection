# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "home"
__date__ = "$27 Apr, 2021 7:57:27 PM$"

import pandas as pd
from sklearn.cluster import KMeans
from sqlalchemy import create_engine
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
import numpy as objnumpy
from sklearn import metrics

password = ''

db_connection_str = 'mysql+pymysql://root:' + password +'@localhost/anomalydetection?charset=utf8'
db_connection = create_engine(db_connection_str)

df = pd.read_sql('SELECT * FROM kdddataset', con=db_connection)

# you want all rows, and the feature_cols' columns
X = df.iloc[ : , 8 : 42].values
y = df.iloc[ : , 4 : 5].values

print('X Data::', X)

kmeans = KMeans(n_clusters = 4)
kmeans.fit(X)

y_kmeans = kmeans.predict(X)

print("K-Means Accuracy :", metrics.accuracy_score(y, y_kmeans)); 

# Split into training and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42) 

knn = KNeighborsClassifier(n_neighbors=7) 

knn.fit(X_train, y_train) 

# Predict on dataset which model has not seen before 
y_knn = knn.predict(X_test);

print("KNN Accuracy :", metrics.accuracy_score(y_test, y_knn)); 
