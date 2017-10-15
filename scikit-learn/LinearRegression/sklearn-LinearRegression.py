'''
Created on Oct 16, 2016

@author: Cheng-lin Li
'''
from __future__ import print_function
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression


def getInputData(filename):
    _data = np.genfromtxt(filename, delimiter = ',')
    _X = _data[:, :2]
    _Z = _data[:, 2]
    return _X, _Z
    
def plot(X, Z, lr):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
     
    x = X[:, 0]
    y = X[:, 1]
    z = np.array(Z[:])
     
    ax.scatter(x, y, z, c='r', marker='o')
    

    xx, yy = np.meshgrid(np.arange(x.min()-0.2, x.max()+0.2, 0.02), np.arange(y.min()-0.2, y.max()+0.2, 0.02))
    zz = np.zeros(shape = (xx.shape))
    for i in range(len(xx)):
        for j in range(len(xx[i])):
            zz[i][j] = lr.predict([[xx[i][j], yy[i][j]]])

    ax.plot_surface(xx, yy, zz, color = 'b', alpha = 0.3) # Plot the segmentation plane

    plt.show()
        

if __name__ == '__main__':
    
    X, Z = getInputData("linear-regression.txt") #Get column 1,2 as X, column 3 as Z
    lr = LinearRegression()
    lr.fit(X, Z)
    print(str(lr.get_params))
    plot(X, Z, lr)


          