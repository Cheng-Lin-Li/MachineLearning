## This is an implementation of Hierarchical Agglomerative Clustering (HAC) algorithm in Python 3.


## The task:
This task implements document clustering algorithm by hierarchical (agglomerative/bottom-up) clustering HAC in Python 3.

In this implementation, represent each document as a unit vector. The unit vector of a document is obtained from tf*idf vector of the document, normalized (divided) by its Euclidean length. 

The tf is term frequency (# of occurrences of word in the document) and idf is given by log((N+1)/(df+1)), where N is the number of documents in the given collection and df is the number of documents where the word appears.

Use log of base 2 to compute idf.

Use Cosine function A*B/(‖A‖‖B‖) to measure their similarity/distance.

Use the distance between centroids to measure the distance of two clusters in hierarchical clustering.
In the algorithm, the centroid is given by taking the average over the unit vectors in the cluster. 
It does not need to normalize the centroid.
The algorithm returns the clusters found for a desired number of clusters.

#### Usage: python HAC.py docword.enron_s.txt k
The program takes 2 arguments:
  1. docword.txt is a document-word file. The format will be listed in the input section.
  2. k is the desired number of clusters.

#### Input: docword.enron_s.txt
The data file is a Bag-of-words dataset (https://archive.ics.uci.edu/ml/datasets/Bag+of+Words) at UCI Machine Learning Repository to test the algorithms.
The data set contain five collections of documents. Documents have been pre-processed and each collection consists of two files: vocabulary file and document-word file. This implementation will only use document-word file.

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
The third line (3710420) is the number of words that appear in at least one document.
Or you can treat it as total number of records.

Starting from the fourth line, the content is <document id> <word id> <tf>. 
For example, document #1 has word #118 (i.e., the line number in the vocabulary file) that occurs once.

#### Output: The documents IDs for each cluster.
The output displays into standard output.
For each cluster, output documents IDs that belong to this cluster in one line. Each ID is separated by comma. For example,

96,50

79,86,93 97

4,65,69,70

...

The order doesn’t matter

## Data Structure
  1. Since the number of words in the vocabulary may be very big, The implementation uses csc_matrix to store and operate unit vectors.
  
  2. Reference of csc matrix on https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html#scipy.spars e.csc_matrix

## Process

  1. The implementation use heap-based priority queue to store pairwise distances of clusters. Python heapq module (https://docs.python.org/3.6/library/heapq.html) was included to implement the heap. 

  2. The implementation uses delayed deletion for the removal of nodes that involve the clusters which have been merged to improve efficiency.
    
    a. Instead of removing old nodes from heap which will take O(n^2) where n is the number of data points to be clustered, the implementation just keeps these nodes in the heap.
    b. Every time the program pop up an element from the heap, check if this element is valid (does not contain old clusters) or not. If it is invalid, continue to pop another one.
