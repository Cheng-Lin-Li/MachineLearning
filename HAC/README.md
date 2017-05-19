## This is an implementation of  (HAC) in Python 3.


## The task:
Use PCA to reduce the dimensionality of the data points in pca-data.txt from 3D to 2D. Each line of the data file represents the 3D coordinates of a single point. 

Program will output the directions of the first two principal components.


#### Usage: python PCA.py


#### Input: pca-data.txt
The data file that contains the relevant records.
The file name and target dimensions were coded at main section in program but you can easily replace the code by your own to read the input file from parameters.

#### Output: Sorted K eigenvector and the results that reduces to k dimensions


## Data Structure
  1.	All data points store in an array.
   
    a.	The data structure is [point 0, point 1, …, point N-1]
    b.	For each point, the data structure is [x, y, z], in which x, y and z stand for the three coordinates respectively of points
    c.	mn_x is the data set after mean normalization.
  2.	Covariance stores in an array.
  
    a.	covar: Store covariance of all data points, which is a n*n matrix.
  3.	Eigenvactor stores in an array.
  
    a.	Store covariance of all data points, which is a n*n matrix.
    b.	eigenvector = [(eigenvector 1), (eigenvector 2), …, (eigenvector n)]. n is the dimension of data points.
    c.	sorted_eigenvector = [(eigenvector 1), (eigenvector 2), …, (eigenvector n)]
    d.	sorted_k_eigenvector = [(eigenvector 1), (eigenvector 2), …, (eigenvector k)]. k is the dimension we want to reduce to.
  4.	Eigenvalue stores in an array.
  
    a.	Store weighted values of different eigenvector. covariance of all data points, which is a n*n matrix.



## Process

  1.	Assign k as the number of dimension that want to reduce to.
  2.	Get all data points from input.
  3.	Normalize data point: calculate the mean value μ of all data points and assign (X(i) – μ) to X(i). (X(i)  stands for point i)
  4.	Compute covariance matrix Σ of all data points.
  5.	Compute eigenvalue & eigenvectors of matrix Σ, which is denoted as S and U.
  6.	Take first k column S and corresponding of U matrix as Ureduce.
  7.	Calculate the k-dimension representation of each point X(i) with Z(i) = Ureduce * X(i).
  8.	Print and plot the result


