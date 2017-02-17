#!/usr/bin/env python
# encoding: utf-8
'''
Machine Learning Algorithm Name: K means

This is a sample program to demonstrate the implementation of K means

@author: Jianfa Lin
@revise: Cheng-Lin Li a.k.a. Clark

@copyright:  2017 Cheng-Lin Li@University of Southern California. All rights reserved.

@license:    Licensed under the GNU v3.0. https://www.gnu.org/licenses/gpl.html

@contact:    jianfali@usc.edu, clark.cl.li@gmail.com
@version:    2.1

@create:    September, 25, 2016
@updated:   February, 16, 2017
'''

# Procedure:

# Randomly generate K point for initial centroid
# while centroid movement is 0 or recursion reach time reachs a given value
# for i=1 to m get c(i)
# for k=1 to K move centroid
import numpy as np
import random
import math
from scipy.cluster.hierarchy import centroid

K = 3  # number of cluster
CONVERGENCE = 1e-9 # The recursive termination threshold of total of square of ||xi - uk||

def get_data (fn):
    X = np.genfromtxt(fn, delimiter=',')
    return X
    
class kmeans:
    def __init__(self, data=[], k=0, convergence = CONVERGENCE ):
        self.datapoints = data  # Data points array
        self.size = len(data)
        self.data_classification = [-1 for x in range(self.size)] #Store the classification of each point
        self.K = k  # numbers of cluster
        self.conv = convergence
        self.centroid = []
        self.new_centroid = [] # Initialize new_centroid
        self.current_distance = []
        self.weight = []
        self.move_stop = False
        self.t = 0  # Count the recursion time
        self.dim = 0
        self.new_distance=[]
        if len(data) > 0 :
            self. execute(self.datapoints, self.K)
                
    def execute(self, x, k):
        self.datapoints = x
        self.K = k
        self.size = len(x)
        self._initial()
        
        while not self.move_stop :
            self.new_centroid = []
            for x in range(0, k):
                self.new_centroid.append([0, 0, 0])
                
            self._calculate_distance()
            self._move_step()
            self._terminate()
               
        print('Recursion time:', self.t)
        print('Final centroids:', self.centroid)
        print('Weight:',self.weight)       

    def _initial(self):
    # Randomly generate K points as initial centroids
        for i in range(0, self.K):  
            self.centroid.append(self.datapoints[int(random.random()*(self.size-1))].tolist())
                # Initialize new_centroid
        print ('Initial centroids:'+ str(self.centroid))
        
        self.size = len(self.datapoints)
        self.dim = self.datapoints.ndim
        self.new_centroid = [[0 for x in range(self.dim+1)] for y in range (self.K)]
        self.data_classification = [-1 for x in range(self.size)]
        self.weight = [0 for x in range(self.K)]
        
    def _calculate_distance(self):
    # Cluster step
        self.current_distance = []
        current_distance = self.current_distance
        datapoints = self.datapoints
        centroid = self.centroid
        data_classification = self.data_classification
        new_centroid = self.new_centroid
        K = self.K
        size = self.size
        weight = self.weight
        
        for idx_p, point in enumerate(datapoints):  # Cluster the datapoints in K groups
            shortest = 10000
            for index, cpoint in enumerate(centroid):
                distance = math.pow(point[0] - cpoint[0], 2) + math.pow(point[1] - cpoint[1], 2)  # Calculate the Euclidian distance between the point and every centroid
                if (distance < shortest):  # Try to find the shortest distance from one of K centroids
                    shortest = distance
                    self.data_classification[idx_p] = index  # Indicate to which cluster/centroid the point is belong
        
            new_centroid[data_classification[idx_p]][0] += point[0]  # Sum up the x coordinate values of all datapoints in a same cluster
            new_centroid[data_classification[idx_p]][1] += point[1]  # Sum up the y coordinate values of all datapoints in a same cluster
            new_centroid[data_classification[idx_p]][2] += 1         # Record the number of datapoints in a cluster
            current_distance.append(shortest)      # In order to calculate the total of square of ||xi - uk||
            
        for i in range(K):
            weight[i] = new_centroid[i][2] / size                
    
    def _move_step(self):
        self.new_distance = []
        new_distance = self.new_distance
        datapoints = self.datapoints
        centroid = self.centroid
        new_centroid = self.new_centroid
        data_classification = self.data_classification
        
        for index, cpoint in enumerate(centroid):
            if new_centroid[index][2] != 0:
                new_centroid[index][0] = round(new_centroid[index][0] / new_centroid[index][2], 9)  # Calculate the average x coordinate value for new centroid
                new_centroid[index][1] = round(new_centroid[index][1] / new_centroid[index][2], 9)  # Calculate the average y coordinate value for new centroid
                distance = 0  # The total of square of ||xi - uk|| with new centroids
                for idx_p, point in enumerate(datapoints):
                    if data_classification[idx_p] == index:
                        distance += math.pow(point[0] - new_centroid[index][0], 2) + math.pow(point[1] - new_centroid[index][1], 2)
                        
                new_distance.append(distance)
        
    def _terminate(self):
        # Have made optimization on condition for stop moving
        # If the total Euclidian distance between centroids and their datapoints is 1e-9 (0.000000001) larger than that before moving centroid,
        # then move the centroid.
        # Else, stop moving centroid and reach a local minimum.
        new_distance = self.new_distance
        current_distance = self.current_distance
        new_centroid = self.new_centroid
        centroid = self.centroid
        
        if abs(sum(new_distance) - sum(current_distance)) > self.conv:  # Minimize the total of square of ||xi - uk|| until reaching threshold
        # if t <= 50:
            for index in range(0, self.K):
                centroid[index][0] = round(new_centroid[index][0], 9)
                centroid[index][1] = round(new_centroid[index][1], 9)
                print ('new_distance:'+ str(sum(new_distance))+ ', current_distance:'+ str(sum(current_distance)))
        else:
            print ('new_distance:'+ str(sum(new_distance))+ ', current_distance:'+ str(sum(current_distance)))
            self.move_stop = True
                
        self.t += 1


if __name__ == '__main__':
    print ('This program execute\n')
    data_set = get_data ('clusters.txt')
    kms = kmeans(data_set, 3, 1e-9)
    
else:
    pass
    #print ('The code is imported from another module\n')