#!/usr/bin/env python
# encoding: utf-8
'''
Hierarchical Agglomerative Clustering (HAC) algorithm -- An implementation of bottom-up clustering algorithm by heap-based priority queue.

The implementation goal is to clustering documents into groups.
Each document will be represented as a unit vector, the unit vector of a document is obtained from tf*idf vector of the document, normalized (divided) by its Euclidean length.
tf is term frequency (# of occurrences of word in the document) and idf is given by log[(N+1)/(df+1)], where N is the number of documents in the given collection and df is the number of documents where the word appears.

Use Cosine function (A*B)/(||A|| ||B||) to measure their similarity/distance.
Use the distance between centroids to measure the distance of two clusters in hierarchical clustering.
In the algorithm, the centroid is given by taking the average over the unit vectors in the cluster.
   
    Major Functions:
    
@author: Cheng-Lin Li a.k.a. Clark Li

@copyright:    2017 Cheng-Lin Li@University of Southern California. All rights reserved.

@license:    Licensed under the GNU v3.0. https://www.gnu.org/licenses/gpl.html

@contact:    chenglil@usc.edu or clark.cl.li@gmail.com
@version:    1.0

@create:    April, 3, 2017
@updated:   April, 14, 2017
'''

from __future__ import print_function 
from __future__ import division

import collections
import itertools
import sys
import heapq
import numpy as np
from scipy.sparse import csc_matrix
from datetime import datetime

__all__ = []
__version__ = 1.0
__date__ = '2017-04-03'
__updated__ = '2017-04-14'

SPLITTER = ' ' #Data separate symbol
DATA_INDEX_FROM = 1 #Data index from 0 or 1. example M(0,0)=2 or M(1,1)=2. The system default index is 0
UNIT_VECTOR = 'tf-idf' # Unit vector of a document. Currently only support tf-idf
DISTANCE_FUNCTION = 'cosine_similarity' # Distance function. Currently only support tf-idf

USE_UNICODE = False
DEBUG = 0 # Level:0=No log, :1=Normal, :2=Detail
PRINT_TIME = False #True/False to enable/disable time stamp printing into result respectively. 

INPUT_FILE = 'input.txt' #Default input file name
ORIG_STDOUT = None
#OUTPUT_FILE = 'output.txt' # OUTPUT_FILE COULD BE 'OUTPUT_FILE = None' for console or file name (e.g. 'OUTPUT_FILE = 'output.txt') for file.'
OUTPUT_FILE = None # OUTPUT_FILE COULD BE 'OUTPUT_FILE = None' for console or file name (e.g. 'OUTPUT_FILE = 'output.txt') for file.'


def getInputData(filename):
# Get data from input file. 
    _row = list()
    _col = list()
    _data = list()
    _no_docs = 0
    _counter = 0
    
    try:
        with open(filename, 'r') as _fp:
            for _each_line in _fp:
                if _counter >= 3: #skip file header
                    _r = _each_line.strip().split(SPLITTER)
                    _row.append(int(_r[0])-DATA_INDEX_FROM)
                    _col.append (int(_r[1])-DATA_INDEX_FROM)
                    _data.append(float(_r[2])) #(data=tf, indices=document id, indptr=word id)
                elif _counter == 0:
                    _no_docs = int (_each_line)
                    _counter += 1
                else:
                    _counter += 1
        _fp.close()
        if DEBUG: print ('getInputData.=>no. of documents=%d'%(_no_docs))
        if DEBUG: print ('getInputData. row = : %s'%(_row))
        if DEBUG: print ('getInputData. col = : %s'%(_col))
        if DEBUG: print ('getInputData. data = : %s'%(_data))
        return _no_docs, _row, _col, _data
    except IOError as _err:
        print ('File error: ' + str (_err))
        exit()

def set_std_out_2_file(filename):
    try:
        ORIG_STDOUT = sys.stdout        
        if filename != None :
            f = file(filename, 'w')
            sys.stdout = f
        else:
            pass    
    except IOError as _err:
        if (DEBUG == True): 
            print ('File error: ' + str (_err))
        else :
            pass
        exit()
        
def restore_std_out():
    try:
        sys.stdout.flush()
        sys.stdout.close()
        sys.stdout = ORIG_STDOUT                         
    except IOError as _err:
        if (DEBUG == True): 
            print ('File error: ' + str (_err))
        else :
            pass
        exit()
    
def setOutputData(filename='', result_dict=dict()):
# output results. 
    try:
        if filename != None :
            orig_stdout = sys.stdout
            f = file(filename, 'w')
            sys.stdout = f
        else:
            pass
##########  
        for key in result_dict:
            print (','.join(map(lambda x: str(x+DATA_INDEX_FROM), key)))

###########
        sys.stdout.flush()       
        if filename != None :
            sys.stdout = orig_stdout                   
            f.close()
        else:
            pass        
    except IOError as _err:
        if (DEBUG == True): 
            print ('File error: ' + str (_err))
        else :
            pass
        exit()
        
class priority_queue(object):
    # This is a priority queue implementation which introduce from Python official website. 
    # https://docs.python.org/2.7/library/heapq.html
    def __init__(self):    
        self.pq = []                         # list of entries arranged in a heap
        self.entry_finder = {}               # mapping of tasks to entries
        self.REMOVED = '<removed-task>'      # placeholder for a removed task
        self.counter = itertools.count()     # unique sequence count

    def add_task(self, task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heapq.heappush(self.pq, entry)

    def remove_task(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        if DEBUG > 1: print('priority_queue.remove_task=%s'%(str(task)))
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop_task(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            _priority, _count, task = heapq.heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')
            

class HAC(object):
    '''
    This class implements Hierarchical Agglomerative Clustering (HAC) algorithm.
    Major functions:
        execution: input number of documents, document matrix as a CSC matrix format, clustering number k
            - CSC matrix is the standard Compressed Sparse Column matrix (CSC) representation where indicates by three parts:
            a. The row indices for column i are stored in indices[indptr[i]:indptr[i+1]] and 
            b. Their corresponding values are stored in data[indptr[i]:indptr[i+1]].
            c. The index pointer (indptr) points out how many data should be stored in the specific column (indices).
            d. The length of indices and data should be the same.
          
    '''   
    hash_func = None

    def __init__(self, unit_vector=UNIT_VECTOR, distance_function=DISTANCE_FUNCTION):
        '''
        Constructor
            self.results: The clusters result stores in a dictionary.
                Key: a tuple for cluster member nodes. (0, 5, 7, 10,...)
                value: a tuple to store (centroid vector, sum of vector, total number of cluster members. )
        '''
        self.results = dict()
        self.no_docs = 0
        self.k = 0
        self.data_matrix = None
        if unit_vector == 'tf-idf':
            self.unit_vector = self.tf_idf
        else:
            return None        
        if distance_function == 'cosine_similarity':
            self.distance_function = self.cosine_similarity
        else:
            return None

        
    def execution(self, no_of_docs, document_matrix, k):
        self.no_docs = no_of_docs
        self.data_matrix = csc_matrix(document_matrix)
        self.data_matrix = self.get_unit_vector(self.data_matrix, self.unit_vector)
        self.results = self.get_clusters(self.data_matrix, k, self.distance_function)
        if DEBUG: print ('self.results=%s'%(str(self.results)))
        return self.results
    
    def get_clusters(self, data_matrix, k, distance_function):
#       results: The clusters result stores in a dictionary.
#           Key: a tuple for cluster member nodes. (0, 5, 7, 10,...)
#           Value: a tuple to store (centroid vector, sum of vector, total number of cluster members. )
                
        _result = dict()
        _pq = priority_queue()
        _task = None
        _centroid = None       
        _new_cluster = None 
        # Initial clusters for each individual document
        for _i in range(self.no_docs):
            _result[tuple([_i])] = [data_matrix.getrow(_i), 1, data_matrix.getrow(_i)] # store [vector, total no. of document, sum of vectors]
        _result = collections.OrderedDict(sorted(_result.items()))
        if DEBUG > 1: print('HAC.get_clusters=>=>_result=%s'%(str(_result)))
        
        # for K clusters
        while len(_result) > k:
            if DEBUG: print('HAC.get_clusters=>number of clusters = len(_result) = %d'%(len(_result))) 
            _key_list = _result.keys() #Get all keys/nodes/clusters from _result dictionary
            self.set_new_clusters(_result, _pq, _new_cluster, _centroid)
            if DEBUG: print('HAC.get_clusters=>Before _pq.pop_task()')                    
            _task = _pq.pop_task()
            if DEBUG: print('HAC.get_clusters=>get new cluster pairs =>_task=%s'%(str(_task)))
            _v0 = tuple(_task[0]) 
            _v1 = tuple(_task[1])
            _total_documents = _result[_v0][1]+_result[_v1][1]            
            _sum_vectors = _result[_v0][2] + _result[_v1][2]
            _centroid = _sum_vectors/_total_documents # v0 + v1 / total documents

            if DEBUG>1: print('_v0= %s,'%(str(_v0)))    
            if DEBUG>1: print('_v0 sum vectors= %s,'%(str(_result[_v0][2])))
            if DEBUG>1: print('_v1= %s,'%(str(_v1)))
            if DEBUG>1: print('_v1 sum vectors= %s,'%(str(_result[_v1][2])))
            if DEBUG>1: print('v0+v1=_sum_vectors= %s,'%(str(_sum_vectors)))       
            if DEBUG>1: print('_total_documents= %s,'%(str(_total_documents)))                                       
            if DEBUG>1: print('_centroid= %s, _total_documents=%d'%(_centroid, _total_documents))
            # Remove old documents from result clusters.
            self.get_removed_clusters(_result, _pq, _v0)
            self.get_removed_clusters(_result, _pq, _v1)
            
            # Add new cluster into result clusters.
            _result[tuple(sorted(itertools.chain.from_iterable((_task[0], _task[1]))))]=[_centroid, _total_documents, _sum_vectors] 
            _new_cluster = tuple(sorted(itertools.chain.from_iterable((_task[0], _task[1]))))
            if DEBUG: print('HAC.get_clusters=>_new_cluster=%s'%(str(_new_cluster)))
            if DEBUG > 1:
                for key in _result:
                    print (','.join(map(lambda x: str(x), key)))
        return _result
    
    def set_new_clusters(self, clusters, priority_queue, new_cluster, centroid ):
        # Calculate new distances between the centroid of new cluster and rest of nodes/documents.   
        _key_list = list(clusters.keys()) #Get all keys/nodes/clusters from _result dictionary
        if len(_key_list) == self.no_docs: # If it is first time to build up clusters                                           
            for _i in range(len(_key_list)): # bottom up clustering                   
                for _j in range(_i+1, len(_key_list)):
                    _v0 = clusters[_key_list[_i]][0]
                    _v1 = clusters[_key_list[_j]][0]
                    _t = tuple((_key_list[_i], _key_list[_j]))
                    if DEBUG>1: print('HAC.set_new_clusters=> new cluster =%s'%(str(_t)))
                    _d = self.get_distance(_v0, _v1, self.distance_function)
                    priority_queue.add_task(_t, -1*_d)
        else:       
            for _ki in clusters: # Pair for new clusters with ascending order
                _isSamePair = False
                if len(_ki) < len(new_cluster): #new cluster = (small nodes cluster, large nodes cluster)
                    _t = tuple((_ki, new_cluster))
                elif len(_ki) == len(new_cluster): # if two clusters have same number of nodes, then compare the sequence of nodes
                    if _ki < new_cluster:
                        _t = tuple((_ki, new_cluster))
                    elif _ki > new_cluster:
                        _t = tuple((new_cluster, _ki))
                    else:
                        _isSamePair = True
                else:
                    _t = tuple((new_cluster, _ki))
                    
                if _isSamePair == False:
                    if DEBUG: print('HAC.set_new_clusters=> new cluster =%s'%(str(_t)))
                    _d = self.get_distance(clusters[_ki][0], centroid, self.distance_function)                    
                    priority_queue.add_task(_t, -1*_d)
                
    def get_removed_clusters(self, clusters, priority_queue, element):
        # Remove old documents from result clusters.
        del clusters[element]
        # Remove documents that include in new cluster from priority queue / heap
        for _ki in clusters:
            if len(_ki) < len(element):
                _t = tuple((_ki, element))
            elif len(_ki) == len(element):
                if _ki < element:
                    _t = tuple((_ki, element))
                else:
                    _t = tuple((element, _ki))
            else:
                _t = tuple((element, _ki))
            try:
                priority_queue.remove_task(_t)        
            except KeyError as ke:
                if DEBUG : print('HAC.get_removed_clusters=>priority_queue.remove_task: KeyError %s'%(ke))
    
    def get_distance(self, vector1, vector2, distance_function):
        return distance_function(vector1, vector2)
    
    def cosine_similarity(self, vector1, vector2):
        _v1_sqrt = vector1.power(2)
        _v2_sqrt = vector2.power(2)
        _v1 = vector1.todense()
        _v2 = vector2.todense()
        if DEBUG > 0: print('HAC.cosine=>v1 centroid=%s, v2 centroid=%s, cosine()=%s'%(str(vector1), str(vector2), _v1.dot(_v2.transpose())/(np.sqrt(_v1_sqrt.sum())*np.sqrt(_v2_sqrt.sum())))) 
        return (_v1.dot(_v2.transpose())/(np.sqrt(_v1_sqrt.sum())*np.sqrt(_v2_sqrt.sum())))
    
    def get_unit_vector(self, document_matrix, unit_vector):
        return unit_vector(document_matrix)
            
    def tf_idf(self, data_csc_matrix):
        _tf_idf = list()
        _pre_ipt = 0
        if DEBUG: print('tf_idf=> input data_vectors=%s'%(data_csc_matrix.todense()))
        data_csc_matrix.eliminate_zeros() #Remove zeros from the matrix
        df_array = data_csc_matrix.getnnz(axis=0)    # Number of stored values, including explicit zeros.
        if DEBUG: print('tf_idf=> df is the number of documents where the word appears., df_array=%s'%(str(df_array)))
        
        # Calculate weighting=tf*idf vector for each document.
        # This is column/word based calculation.
        _data = data_csc_matrix.data
        _indices = data_csc_matrix.indices
        _indptr = data_csc_matrix.indptr
        if DEBUG>1: print('tf_idf=> _data=%s'%(str(_data)))
        if DEBUG>1: print('tf_idf=> _indices=%s'%(str(_indices)))
        for _i in range(len(_indptr)-1): # Calculate from each column
            _pre_ipt = _indptr[_i] # previous ipt (index pointer of the CSC matrix)
            _ipt = _indptr[_i+1]
            if _ipt == _pre_ipt: # if previous index pointer is the same as current pointer, no data in this column
                pass # Escape the column which is no element.
            else:
                for _j in range(_pre_ipt, _ipt):
                    _data[_j] = _data[_j]*np.log2((self.no_docs+1)/(df_array[_i]+1))
                    if DEBUG>1: print('_j=%s, self.no_docs+1=%d, df=%f, tf*idf=%f'%(_j, self.no_docs+1, df_array[_i]+1, _data[_j]))
            if DEBUG>1: print('np.log2((self.no_docs+1)/(df_array[_i]+1))=%f'%(np.log2((self.no_docs+1)/(df_array[_i]+1))))
            if DEBUG>1: print('_data=%s'%(_data))
        
        data_csc_matrix.data = _data # For easy understandable purpose, actually not necessary to re-assign it again because we operate all data based on the _data pointer.
        if DEBUG>1: print('tf_idf=>data_csc_matrix.todense()=%s'%(data_csc_matrix.todense()))
        if DEBUG: print('tf_idf=>data_csc_matrix.getrow(0)=%s'%(data_csc_matrix.getrow(0)))
        if DEBUG: print('tf_idf=>data_csc_matrix.getrow(1)=%s'%(data_csc_matrix.getrow(1))) 
        
        #
        # Normalization.
        #    Normalized (divided) by its Euclidean length.
        #    Euclidean distance for each document = sqrt(w1^2+w2^2+...)
        data_csc_matrix_sqrt = data_csc_matrix.power(2)
        data_csr_matrix = data_csc_matrix.tocsr() # Transfer to CSR for normalization
        _data = data_csr_matrix.data
        _indices = data_csr_matrix.indices
        _indptr = data_csr_matrix.indptr   
        if DEBUG: print('tf_idf=>data_csr_matrix.data=%s'%(data_csr_matrix.data))     
        for _i in range(len(_indptr)-1): # Calculate from each row
            _pre_ipt = _indptr[_i] # previous ipt (index pointer of the CSR matrix)
            _ipt = _indptr[_i+1]
            _ed = np.sqrt(data_csc_matrix_sqrt.getrow(_i).sum()) #Calculate Euclidean distance
            if _ipt == _pre_ipt:
                pass# Escape the Row which is no element.
            else:
                for _j in range(_pre_ipt, _ipt):
                    if DEBUG > 1: print ('tf_idf=>_data[_j]=%f, _ed=%f, _data[_j]/_ed=%f'%(_data[_j], _ed, _data[_j]/_ed))
                    _data[_j] = _data[_j]/_ed
        data_csr_matrix.data = _data
        data_csc_matrix = data_csr_matrix.tocsc()
        if DEBUG: print('tf_idf=>Normalization: data_csc_matrix.getrow(0)=%s'%(data_csc_matrix.getrow(0)))
        if DEBUG: print('tf_idf=>Normalization: data_csc_matrix.getrow(1)=%s'%(data_csc_matrix.getrow(1)))     
        
        return data_csc_matrix

if __name__ == "__main__":
    '''
        Main program.
    '''
    output_file = ''
    if len(sys.argv) < 3 : 
        print('Usage of Hierarchical Agglomerative Clustering (HAC) algorithm: %s inputFile.txt k [output.txt]'%(sys.argv[0]))
        print('    inputFile.txt is a document-word file.')
        print('    k is the number of initial points we get.')
        print('    output.txt is the output file. It is an option parameter')
        exit()
    else:
        input_file = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE
        k = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        output_file = sys.argv[3] if len(sys.argv) > 3 else OUTPUT_FILE
    if PRINT_TIME : print ('HAC=>Start=>%s'%(str(datetime.now())))   
    
    if output_file != None: set_std_out_2_file(output_file)

    # Initial HAC object and read input file
    _no_of_docs, _row, _col, _data = getInputData(input_file)
    _document_matrix = csc_matrix((_data, (_row, _col)), dtype=np.float64)
    if DEBUG: print('Main._document_matrix=%s'%(_document_matrix))
    hac = HAC()
    _results = hac.execution(_no_of_docs, _document_matrix, k)
    setOutputData(output_file, _results)
    
    if output_file != None: restore_std_out()
    if PRINT_TIME : print ('HAC=>Finish=>%s'%(str(datetime.now())))   
    
