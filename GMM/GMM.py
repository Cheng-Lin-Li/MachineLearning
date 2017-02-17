#!/usr/bin/env python
# encoding: utf-8
'''
Machine Learning Algorithm Name: Gaussian Mixture Model (GMM)

This is a sample program to demonstrate the implementation of Gaussian Mixture Model (GMM) by Expectation Maximization (EM) algorithm

@author:    Jianfa Lin
@revise:    Cheng-Lin Li a.k.a. Clark

@copyright:  2017 Cheng-Lin Li@University of Southern California. All rights reserved.

@license:    Licensed under the GNU v3.0. https://www.gnu.org/licenses/gpl.html

@contact:    jianfali@usc.edu, clark.cl.li@gmail.com
@version:    1.6

@create:    September, 29, 2016
@updated:   February, 16, 2017
'''

import numpy as np
import random
import math

cluster_num = 3  # Number of Gaussian distributions
givenCentroids = np.array([[3.083182557, 1.776213738], [-0.974765718, -0.684193041], [5.620165735, 5.026226344]])
Alphas = np.array([31/150, 85/150, 34/150])
Sigmas = np.array([[[1.0,0.0],[0.0,1.0]], [[1.0,0.0],[0.0,1.0]], [[1.0,0.0],[0.0,1.0]]])
LIKELIHOOD = -10000.0
Mus = givenCentroids


def getInputData(filename):
    # Get data from data file
    training_data = np.genfromtxt(filename, delimiter=',')
    return training_data


def initCentroids(K, data):
    # Select K points from datapoints randomly as centroids
    N = len(data)
    index = random.sample(range(N), K)
    dim = data.shape[1]  # Get the dimension of datapoint
    centroids = np.zeros((K, dim))
    for i, j in enumerate(index):
        centroids[i] = data[j]

    return centroids


def assignCentroids():
    # Assign given values to centroids
    centroids = givenCentroids
    return centroids

    
'''
GMM_Machine class
Used to implement GMM algorithm
'''
class GMM_Machine():
    def __init__(self, K, data, alpha=Alphas, mu=Mus, sigma=Sigmas, likelihood=LIKELIHOOD):
        self.K = K                        # Number of clusters
        self.data = np.array(data)        # Dataset
        self.N = len(data)                # Length of dataset
        self.r = np.zeros([K,len(data)])  # Weight of each point in different distribution
        self.alpha = np.array(alpha)      # K*1 array
        self.mu = np.array(mu)            # K*d array, in this case d=2
        self.sigma = np.array(sigma)      # K*d*d array, in this case d=2
        self.likelihood = likelihood
        
    def Normal(self, Xi, Uk, Sk, d):
        # Calculate the value for Xi in normal distribution k
        # Xi stands for data[i]
        # Uk stands for mu[k]
        # Sk stands for sigma[k]
        # d stands for the dimension of datapoint
        probability = 1/pow((2*math.pi), -d/2) * pow(abs(np.linalg.det(Sk)), -1/2) * \
                    np.exp(-1/2 * np.dot(np.dot((Xi-Uk).T, np.linalg.inv(Sk)), (Xi-Uk)))
        return probability
    
    def maximizeLLH(self):
        # Calculate the maximum likelihood

        new_likelihood = 0
#        for i in range(self.N):
        for i in range(self.N):
            temp = 0
            for k in range(self.K):
                temp += self.alpha[k] * self.Normal(self.data[i].T, self.mu[k].T, self.sigma[k], self.data.shape[1])
            new_likelihood += np.log(temp)
            #print('check temp type:',type(temp))
        
        print("New_likelihood:",new_likelihood)
        return new_likelihood
        
    def Estep(self):
        # E step
        print("Enter E step.")
        # Calculate r[k][i], which stands for Rik
        s = np.zeros(self.N)
        for i in range(self.N):
            temp = np.zeros(self.K)  # Temporary array
            # Calculate alpha[k]*N(Xi, Uk, Sk) for each data[i] and the summation of that in all distributions
            for k in range(self.K):
                temp[k] = float(self.alpha[k]) * self.Normal(self.data[i].T, self.mu[k].T, self.sigma[k], self.data.shape[1])
                s[i] += temp[k]
            for k in range(self.K):
                self.r[k][i] = temp[k]/s[i]
#                print("self.r[k][i]=",self.r[k][i])
    
    def Mstep(self):
        #M step
        print("Enter M step.")
        for k in range(self.K):
            # Calculate alpha[k]
            self.alpha[k] = np.sum(self.r[k]) / self.N
    
            # Calculate mu[k]
            total = np.zeros(self.mu.shape[1])
            for i in range(self.N):
                total += self.r[k][i]* self.data[i]
            self.mu[k] = total / np.sum(self.r[k])
            
            # Calculate sigma[k]
            summ = np.zeros([self.data.shape[1], self.data.shape[1]])
            for i in range(self.N):
                if self.data[i].ndim == 1:
                    # In our case, data[i] and mu[i] are in the shape like [x1, x2],
                    # which is actually a 1-dimension array, rather than 2*1 or 1*2 matrix.
                    # So have to reshape it to a 2*1 matrix
                    data_temp = self.data[i].reshape(self.data.shape[1], 1)
                    mu_temp = self.mu[k].reshape(self.mu.shape[1], 1)
                    diff_temp = data_temp - mu_temp
                    summ += self.r[k][i] * np.dot(diff_temp, diff_temp.T)
                else:
                    summ += self.r[k][i] * np.dot(self.data[i]-self.mu[i], (self.data[i]-self.mu[i]).T)
            
#            print("summ =",summ,"; np.sum(self.r[k]) =",np.sum(self.r[k]))
            self.sigma[k] = summ / np.sum(self.r[k])
#            print("sigma[k]=",self.sigma[k])        
            
    def execute(self):
        new_lld = self.maximizeLLH()
        recursion = 0
        while(new_lld - self.likelihood > 1e-3):
            self.likelihood = new_lld
            self.Estep()
            self.Mstep()
            new_lld = self.maximizeLLH()
            recursion += 1
        print("Recursion time:", recursion)


if __name__ == '__main__':
    print ('This program execute\n')
    datapoints = getInputData('clusters.txt')
    gmm = GMM_Machine(cluster_num, datapoints)
    gmm.execute()
    print("The likelihood is:", gmm.likelihood)
    print("The amplitudes are:", gmm.alpha)
    print("The means are:", gmm.mu)
    print("The covariances are:", gmm.sigma)
    
else:
    pass
    #print ('The code is imported from another module\n')