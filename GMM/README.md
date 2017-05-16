## This is an implementation of Expectation Maximization (EM) algorithm for for clustering using a Gaussian Mixture Model (GMM) in Python 3

The original author is Jianfa Lin, my classmate in Viterbi School of Engineer in Data Informatics department at University of Southern California during August to December 2016.
You may contact him via: Jianfa Lin <jianfali@usc.edu>

I based on his implementation with modifications as this result.

## The task: 
Run the algorithm on the data file "clusters.txt" using K, the number of clusters, set to 3. 
Report the mean (centroids), amplitude and covariance matrix of each Gaussian in GMM

#### Usage: python GMM.py	

#### Input: A data file (clusters.txt) that contains 150 2D points. Each row in the file contains the coordinates of a single point.
The target cluster numbers (Number of Gaussian distributions), initial centroids (means), alpha (amplitudes), sigma (covariances), likelihood_threshold can be assigned in global variable. It is easy task to modify the code and get those parameters dynamically load during program execution time.

#### Output: Likelihood during every iteration, The amplitudes, means, and coveriances.

## Data Structure
  0  .	All centroids/means store in a list/array.
  
    a.	There are two sets of centroids: centroids and new_centroids.
    b.	centroid = [centroid point1, …, centroid point K]
    c.	The data structure of each centroid is [x, y]
  1.	All data points store in a list/array.
   
    a.	datapoints = [point 0, point 1, …, point N-1]
    b.	The data structure of each point is [x, y]
      i.	x, y is float number.
  2.	A cluster identification list/array, “c”, to label the cluster of each data point.
  
    a.	[label of point 1, label of point 2, …, label of point N]
    b.	The default value is -1.
  3.	Covariance matrix store in a list/array with K, 2D points.
  
    a.	current_cov matrix = [[x1,y1],…,[xk, yk]]
    b.	new_cov_matrix has same data structure.
  4.	A weight storage list/array, “w”, to record the weight of each cluster.
  
    a.	[weight of cluster 1, weight of cluster 2, …, weight of point K].
    b.	current_weight = [float 1, …, float k]
    c.	new_weight has same data structure.
  5.	The posteriori probability of Xi for each cluster (ric) will store in list/array for each cluster.
  
    a.	Structure are  
        [
        [[r1 in cluster 1], [r2 in cluster 1],…, [rn in cluster 1]], 
        [[r1 in cluster 2], [r2 in cluster 2],…, [rn in cluster 2]], 
        …
        [[r1 in cluster K], [r2 in cluster K],…, [rn in cluster K]]
        ]

## Process
  1.	Get all data points from input file, clusters.txt, into system. The data file contains 150 points of 2 dimensions’ data points.
  2.	Call GMM_Machine(X, cluster_no) to initialize the class. 
  3.	Assume the initial covariance matrix store in a list/array, sigma, is [[[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]] for 3 clusters.
  4.	Base on K-Means results to group data points into K initial clusters.
  5.	Create a weight storage list/array, “w”, to record the weight of each cluster.
  6.	Create a cluster identification list/array, “c”, to label the cluster of each data point.
  
    a.	The default cluster value is the result of K-Means.
  7.	M-step: Calculate the log-likelihood:
  8.	E-step: Calculate the new Gaussian distributions.
  
    a.	Calculate new Gaussian distributions to get mean, and covariance matrix of each Gaussian distribution.
  9.	Calculate the maximum log likelihood base on my Gaussian distributions. 
  10.	Termination:
  
    a.	Condition:
      i.	If the improve of maximum log likelihood is less than threshold (1e-3), we stop the process.
    b.	Else repeat step 7 ~ 9.

