# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 12:55:19 2018

@author: Robert Schuldt
"""

# K-Fold cross validation

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#import working directory
dataset  = pd.read_csv('Social_Network_Ads.csv')

#make matrix of features
X = dataset.iloc[:,[2, 3]].values
# : takes all rows, :-1 takes all colums except for last one

#Now capture depedent variable 
Y = dataset.iloc[:,4].values

#Split dataset into training set and test set
from sklearn.model_selection import train_test_split
#X and Y in the test_split are the individuals arrays. 
#We have an X Array and a YArray I.E matrix, and we want to split them each
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 0) 

#feature scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""
we are applying the scaling because we want to see how earnings impact
purchasing. We have wildly variable earning levels. 

we didn't scale Y because it is binary classifer
"""

#Fit regression to training set
#using linear classifier

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf' , random_state = 0)
classifier.fit(X_train, Y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test) #vector of predictions

#making the Confusion Matrix
from sklearn.metrics import confusion_matrix 
#Class has capital letter, function has no capitals
cm = confusion_matrix(Y_test, y_pred)


"""
This is a better way to evaluate
Using k-fold cross validation
"""

from sklearn.model_selection import cross_val_score

accr = cross_val_score(estimator = classifier, X = X_train,y= Y_train, cv=10)
accr.mean()
accr.std()

#The top left and bottom right are our correct predictions
# We need to see what it looks like, but we will develop a visualization
# that is more effective at communicating 

#Visualizing the Training set results 
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step = 0.01),
                     np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', ))(i), label = j)
plt.title('SVM (Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
#Now do for the actual test set 

X_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step = 0.01),
                     np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', ))(i), label = j)
plt.title('SVM Regression (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
