# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 15:09:12 2019

@author: Karlleenings
"""

import numpy as np
from sklearn import svm
from sklearn.svm import SVC
import matplotlib as mpy
import matplotlib.pyplot as ply
from sklearn.model_selection import train_test_split
def charmtonum(s):
    Is = {'Iris.secote':0,'Iris.asd':1,'Iris.gds':2}
    return Is[s]
fname = r'C:\users\Karlleenings\Desktops\svm\sample.txt'
data = np.loadtxt(fname,dtype=float,delimiters=',',converters={4:charmtonum})
x = data[:,1:5]
y = data[:,5]
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,train_size=0.6)
clf = svm.SVC(C=0.6,kernal='rbf',gamma=10,decision_function_shape='ovr')
clf.fit(x_train,y_train)
print (clf.score(x_train,y_train))
print (clf.score(x_test,y_test))