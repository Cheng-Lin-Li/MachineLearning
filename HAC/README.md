## This is an implementation of Hierarchical Agglomerative Clustering (HAC) algorithm in Python 3.


## The task:
This task implements document clustering algorithm by hierarchical (agglomerative/bottom-up) clustering HAC in Python 3.

#### Usage: python HAC.py docword.enron_s.txt k
The program takes 2 arguments:
  1. docword.txt is a document-word file. The format will be listed in the input section.
  2. k is the desired number of clusters.

#### Input: docword.enron_s.txt
The data file is a Bag-of-words dataset (https://archive.ics.uci.edu/ml/datasets/Bag+of+Words) at UCI Machine Learning Repository to test the algorithms.
The data set contain five collections of documents. Documents have been pre-processed and each collection consists of two files: vocabulary file and document-word file. This implementation will only use documentword file.

For example, “vocab.enron.txt” is the vocabulary file for the Enron email collection which contains a list of words, e.g., “aaa”, “aaas”, etc. “docword.enron_s.txt” is the document-word file, which has the following format:

39861

28102

3710420

1 118 1

1 285 1

1 1229 1

1 1688 1

1 2068 1

...
The first line is the number of documents in the collection (39861). 

The second line is the number of words in the vocabulary (28102).

Note that the vocabulary only contains the words that appear in at least 10 documents.
The third line (3710420) is the number of works that appear in at least one document.

Starting from the fourth line, the content is <document id> <word id> <tf>. 
For example, document #1 has word #118 (i.e., the line number in the vocabulary file) that occurs once.

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


