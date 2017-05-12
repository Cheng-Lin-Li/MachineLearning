## This is an implementation of FastMap Algorithm in Python 3.

## Machine Learning Algorithm: FastMap Algorithm.

## The task:
This program implements FastMap to embed the objects in fastmap-data.txt into a 2D space. 

The first two columns in each line of the data file represent the IDs of the two objects; and the third column indicates the symmetric distance between them. If the farthest pair of objects is ambiguous, the one that includes the smallest object ID will be used. 

The objects listed in fastmap-data.txt are actually the words in fastmap-wordlist.txt (nth word in this list has an ID value of n) and the distances between each pair of objects are the Damerau–Levenshtein distances between them. The program will plot the words onto a 2D plane using the previous FastMap solution and see what it looks like.


## Implementation features:
A FastMap class is implemented and provide a lot of value sets for reference.
    
        self.o_pair = An array, the original data set.
        self.dist = An array, the original destination information between each object
        self.k = An integer, the target dimension
        self.obj_k_d = An array, the result: Object distances scale to k dimension.
        self._col = An integer, current processing dimension
        self._max_dist = A float, keep current max. distance of Oa, Ob in current dimension to speed up the performance.
        self._new_dist_set = An array, the new destination information of current processing dimension between each object
        self._max_Oa = An index information of Oa for max. distance for current dimension.
        self._max_Ob = An index information of Ob for max. distance for current dimension.
        self.obj_set = A set() data type, to store the label of total objects. 


## Process

  1. Assign k as the number of dimension that want to reduce to. 

  2. Get all data points from input. 

  3. Find 2 objects Oa & Ob are farthest apart from each other as pivot objects. 

  4. Project the object Oi on line (Oa, Ob) and calculate the distance from Oi to Oa. 

  5. Consider a (n-1)-dimension hyperplane H that is perpendicular to the line (Oa, Ob) and then project objects on the plane.

  6. Calculate the distance between each points on the new hyperplane H. 

  7. Recursively apply step 3 ~ 6 (k) times for k dimensions. 8. Print and plot the result.
