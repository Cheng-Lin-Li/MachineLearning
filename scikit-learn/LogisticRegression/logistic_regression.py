'''
Created on Oct 16, 2016

@author: Cheng-lin Li
'''
from __future__ import print_function
import numpy as np

from sklearn.linear_model import LogisticRegression

def getInputData(filename):
    _data = np.genfromtxt(filename, delimiter=',')
    _X = _data[:, :3]
    _Y = _data[:, 4]
    return _X, _Y     

if __name__ == '__main__':
    
   
    """
    Logistic Regression
    """
    X,Y = getInputData("classification.txt") #Get column 5 as Y
    X = np.c_[np.ones(len(X)), np.array(X)] # Coordinates vector of points, which is added '1' at first column.
    lr = LogisticRegression()
    info = lr.get_params()
    print (info)    
    lr = lr.fit(X, Y)
    score = lr.score(X, Y)
    W = lr.coef_
    print('score =', score)
    print ('W=', W )    
     
    
    




