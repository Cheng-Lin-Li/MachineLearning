# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 16:36:36 2016

@author: Clark
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

trials = 10 #Trail numbers for each algorithm.
clusters = 3 #Cluster numbers

# Get data points
X = np.genfromtxt(r'.\clusters.txt', delimiter=',')

#Print out the dimensions of data.
print ("X.shape="+str(X.shape))
print ('===================')
print ('     K-Means')
print ('===================')
km = KMeans(n_clusters=clusters, n_init=trials, algorithm ='full')
km.fit(X)

centroids = km.cluster_centers_
labels = km.labels_

plt.xlabel("x-axis") 
plt.ylabel("y-axis") 
# Max 10 clusters can be marked by 5 different colors
colors = 2*["r.", "g.", "c.", "b.", "y."] 

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)

plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=20, linewidths=10)
plt.show()

print ('centroids=', centroids, '\n\n\n')


from sklearn import mixture
print ('===================')
print ('     EM-GMM')
print ('===================')
gmm = mixture.GaussianMixture(n_components=clusters, n_init=trials, covariance_type="full")
gmm.fit(X)
labels = gmm.predict(X)
weights = gmm.weights_
means = gmm.means_
n_cov = gmm.covariances_

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)

plt.scatter(means[:,0], means[:,1], marker='x', s=20, linewidths=10)
plt.show()

print ('GMM weights:', weights)
print ('GMM means:', means)
print ('GMM covars: components=', n_cov)


 