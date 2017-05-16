## This is an implementation of This is an implementation code for K means in Python 3.

The original author is Jianfa Lin, my classmate in Viterbi School of Engineer in Data Informatics department at University of Southern California during August to December 2016.
You may contact him via: Jianfa Lin <jianfali@usc.edu>

I based on his implementation with modifications as the result.

## The task: 
Run the algorithm on the data file "clusters.txt" using K, the number of clusters, set to 3. 
Report the Report the centroid and the weight (number of elements in the cluster/total number of elements) of each cluster.


#### Usage: python K-Means.py

#### Input: A data file (clusters.txt) that contains 150 2D points. Each row in the file contains the coordinates of a single point.
The target cluster numbers, distance threshold can be assigned in global variable. It is easy to modify the code and get those inputs dynamically from program execution parameters.

#### Output: new and current sum of distance between each point to its centroid, recursion times, final centroids and weights for each cluster.

## Data Structure
  0.	All centroids/means store in a list/array.
  
    a.	There are two sets of centroids: centroids and new_centroids.
    b.	centroid = [centroid point1, …, centroid point K]
    c.	The data structure of each centroid is [x, y]
  1.	All data points store in a list/array. 
  
    a.	datapoints = [point 1, point 2, …, point N]
    b.	The data structure of each point is [x, y]
      i.	x, y is float number.
  2.	Distance store in a list/array.
  
    a.	Current distance: Store distance between each point with associated current centroid.
      i.	current_distance = [point 1 distance with associated centroid k, …, point N distance with associate centroid k] 
    b.	New distance: Store total distance for each new centroid.
      i.	new_distance = [Total distance of new centroid 1, …, Total distance of new centroid k] 
  3.	Data Classification in a list/array
  
    a.	An array structure to record cluster for each point.
    b.	data_classification = [cluster of point 1, cluster of point 2, …, cluster of point N]
      i.	The default value of identification number of clusters is -1.
  4.	new_centroid has same data structure as data point but add a third dimensions to record down total number of points belong to this new centroid.
  
    a.	new_centroid = [ [x0, y0, n0], [x1, y1, n1], [x2, y2, n2]]
    b.	n0, n1, n2 = total number of points which identify belonging to these new centroids.

## Process
  1.	According to the required number of clusters, K, program randomly pick up K initial centroids from data set.
  2.	Get all data points from input file, clusters.txt, into system. The data file contains 150 points of 2 dimensions’ data set.
  3.	Calculate the distance for each data point and label the cluster:
  
    a.	Calculate the distance between K centroids and each data point.
      i.	Label the data point to shortest distance of cluster by identification variable “data_classification” in data structure.
      ii.	If the distance is the same, the data point will belong to first shortest centroid.
      iii.	Record down the shortest distance for each point base on exist/old centroid points.
  4.	Calculate the new centroids.
  
    a.	Base on the new groups of data points, calculate new centroids.
    b.	Calculate the total distance for each new centroid with its own group of data points.
  5.	Termination:
  
    a.	Condition:
      i.	If the improve of total distance is less than 1e-9, we stop the process.
    b.	Else repeat step 3 ~ 5.
    
