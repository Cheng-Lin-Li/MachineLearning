#!/usr/bin/env python
# encoding: utf-8
'''
Machine Learning Algorithm Name: Fast Map

This is a sample program to demonstrate the implementation of FastMap

@author: Cheng-Lin Li a.k.a. Clark

@copyright:  2017 Cheng-Lin Li@University of Southern California. All rights reserved.

@license:    Licensed under the GNU v3.0. https://www.gnu.org/licenses/gpl.html

@contact:    clark.cl.li@gmail.com
@version:    1.0

@create:    October, 7, 2016
@updated: February, 15, 2017

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
'''
import numpy as np
import matplotlib.pyplot as plt


def getInputData(filename):
    # Get data from data file and split data into two parts.
    #1. Object pair set: All object relationships store in this set.
    #2. Distance set: All distances are store in this set according to the index of Object pair set.
    _object_pair = np.array
    _distance = np.array
    
    _data = np.genfromtxt(filename, delimiter='\t')
    _object_pair =  np.array(_data[0:, 0:2])
    _distance = np.array(_data[0:,2])
    
    return _object_pair, _distance  

def getObjectName(filename):
    # Get data from data file and split data into two parts.
    #1. Object pair set: All object relationships store in this set.
    #2. Distance set: All distances are store in this set according to the index of Object pair set.
    _label_set = []
    
    with open(filename) as f:
        _label_set = f.read().splitlines()
    
    return _label_set  


class FastMap():
    '''
    classdocs
    
    A FastMap class is implemented and provide a lot of value sets for reference.
    This implementation is base on the reference:
    Faloutsos, Christos, and King-Ip Lin. FastMap: A fast algorithm for indexing, data-mining and visualization of traditional and multimedia datasets. Vol. 24. No. 2. ACM, 1995.

    [Optimized]
    Except first dimension the program select the pivot objects from existing distance information base on reference paper.
    For rest of iterations, the program gets the maximum distance and pivot objects directly from self.max_dist, self.Oa, self.Ob that were computed during the "projection_on_hyper_plane()" step. 
    
    function lists:
    1. set_obj_set(object_pair): Return self.obj_set
        Get object id from data input
    2. choose_distance_objects(object_pair, distance_set): Return _Oa, _Ob, _max_dist
        Choose farthest distance and pair of points (pivot objects) between objects.
    3. get_max_distance (object_pair, distance_set, Obj): Return _farthest_O, _max_distance
        Base on input object, "Obj", get farthest distance object then return.
    4. get_distance (object_pair, distance_set, xi, xj): Return distance(float)
        Get distance from distance set base on matching objects (xi, xj) in object pair set.
    5. calculate_projection_distance(object_pair, distance_set, Oa, Ob): Return self.obj_k_d
        Calculate project distance on the axis of pivot objects.
    6. projection_on_hyper_plane(object_pair, distance_set, xi_set, col): Return _Oa, _Ob, _max_dist
        Project all objects into new hyper plane
    7. execute(object_pair, distance, dimension): Return self.obj_k_d
        Execute FastMap algorithm base on input object pair, distance set, and target dimension.
    8. plot(label_set): 
        Plot diagram base on label_set which ordered by the same sequence of object base on ID.
    '''
    def __init__(self, object_pair=None, distance=None, dimension=0):
        '''
        Constructor
        '''
        self.o_pair = np.array(object_pair)
        self.dist = np.array(distance)
        self.k = int(dimension) #Target dimension
        self.obj_k_d = np.array #Objects project to K dimension results.
        self._col = 0 #Current dimension
        self._max_dist = float(0) #Keep current max. distance to speed up the performance.
        self._new_dist_set = np.array
        self._max_Oa = 0
        self._max_Ob = 0        
        self.obj_set = set() 

    def set_obj_set(self, object_pair):
        '''
         Get object id from data input
        '''
        for _each_pair in object_pair:
            self.obj_set.add(_each_pair[0])
            self.obj_set.add(_each_pair[1]) 
            
        self.obj_k_d = np.zeros((len(self.obj_set), self.k))        
        return self.obj_set
        
    def choose_distance_objects(self, object_pair, distance_set):
        '''
        Choose farthest distance and pair of points (pivot objects) between objects.
            [Optimized]
            Except first dimension the program select the pivot objects from existing distance information base on reference paper.
            For rest of iterations, we get the maximum distance and pivot objects directly from self.max_dist, self.Oa, self.Ob that were computed 
            during the "projection_on_hyper_plane()" step.          
        '''        
        _Oa = 0
        _original_Oa = 0
        _Ob = object_pair[0][0]
        _dist_set = distance_set
        _pre_max_dist = 0
        _max_dist = 0
        _distance = 0
        _b_get_max = False #The boolean flag to identify the program get the max. distance or not.
        
        if (self._col == 0):
            while _b_get_max is False:
                _Oa, _max_dist = self.get_max_distance(object_pair, _dist_set, _Ob)
                if ( _max_dist >= _pre_max_dist and _original_Oa != _Oa and _max_dist != 0):
                    _pre_max_dist = _max_dist
                    _original_Oa = _Ob
                    _Ob = _Oa  
                else:
                    _Oa = _original_Oa
                    _max_dist = _pre_max_dist
                    _b_get_max = True
                                        
                    self._max_Oa = _Oa
                    self._max_Ob = _Ob
                    self._max_dist = _max_dist
        else:
            # Directly get pivot objects and max. distance from the "projection_on_hyper_plane()" step.
            _Oa = self._max_Oa
            _Ob = self._max_Ob
            _max_dist = self._max_dist
                       
        return _Oa, _Ob, _max_dist

    def get_max_distance (self, object_pair, distance_set, Obj):
        '''
        Base on input object, "Obj", get farthest distance object then return 
        '''
        _farthest_O = 0
        _max_distance = 0
        for idx, _each_pair in enumerate(object_pair):
            #Match object in each object pair
            if _each_pair[0] == Obj :
                if distance_set[idx] >= _max_distance :
                    _max_distance = distance_set[idx]
                    _farthest_O = _each_pair[1]
                else:
                    pass
            elif _each_pair[1] == Obj :
                if distance_set[idx] >= _max_distance :
                    _max_distance = distance_set[idx]
                    _farthest_O = _each_pair[0] 
                else:
                    pass                   
            else:
                pass
        
        return _farthest_O, _max_distance
    

    def get_distance(self, object_pair, distance_set, xi, xj):
        '''
        Get distance from distance set base on matching objects (xi, xj) in object pair set.
        '''
        for idx, _each_pair in enumerate(object_pair):
            #Match object in each object pair
            if (xi == xj) :
                return 0 
            elif (_each_pair[0] == xi and _each_pair[1] == xj) or (_each_pair[1] == xi and _each_pair[0] == xj) :
                return distance_set[idx]
            else:
                pass
    
    def calculate_projection_distance(self, object_pair, distance_set, Oa, Ob):
        '''
        Compute the projection distance on pivot objects (Oa, Ob) for each object. 
        '''
        _xi = 0
        _dis_set = distance_set
        _obj_pair = object_pair
        _Dai = 0
        _Dab = self._max_dist
        _Dbi = 0
        
        for _idx, _i in enumerate(self.obj_set):
            # Calculate new distance xi and store as a new dimension into self.obj_k_d             
            _Dai = self.get_distance(_obj_pair, _dis_set, Oa, _i)
            _Dbi = self.get_distance(_obj_pair, _dis_set, Ob, _i)           
            _xi = (_Dai**2 + _Dab**2 - _Dbi**2) / (2 * _Dab)
            self.obj_k_d[_idx][int(self._col)]=_xi
        
        return self.obj_k_d
    
    def projection_on_hyper_plane(self, object_pair, distance_set, xi_set, col):
        '''
        Compute the distance of each object on new hyper plane.
            [Optimized]
            The max. distance, Oa, Ob will be kept during the computation for next iteration.         
        '''
        _xi = 0
        _obj_pair = object_pair
        _dist_set = distance_set
        _new_dist_set =np.zeros((len(_obj_pair), 1)) #New distance set in projected hyper-plane.
        _xi = xi_set
        _col = col
        _D_Oij = 0
        _D_xij = 0
        _max_dist = 0
        _max_Oa = 0
        _max_Ob = 0
                
        for _idx, _pair in enumerate(_obj_pair) :
            _D_Oij = self.get_distance(object_pair, distance_set, _pair[0], _pair[1])
            _D_xij = abs(_xi[int(_pair[0]-1)][int(_col)] - _xi[int(_pair[1]-1)][int(_col)])
            _new_dist_set[_idx] = np.sqrt((_D_Oij**2)-(_D_xij**2))
            
            if (_new_dist_set[_idx] >= _max_dist):
                _max_dist = _new_dist_set[_idx]
                _max_Oa = _pair[0]
                _max_Ob = _pair[1]
                  
        self._new_dist_set = _new_dist_set
        self._max_dist = _max_dist
        self._max_Oa = _max_Oa
        self._max_Ob = _max_Ob
        
        return self._new_dist_set        
    
    
    def execute(self, object_pair, distance, dimension):
        '''
        Execute function to enable the calculation.
        '''
        self.o_pair = object_pair
        self.dist = distance
        self.k = dimension
        _Oa = 0
        _Ob = 0
        _max_dist = 0
        _dist_set = np.array(distance)
        _new_dist_set = np.array
        _col = 0 # temp. variable to point to the column of x array with k dimension.        
        self.obj_set = self.set_obj_set(object_pair)
        
        while (_col < self.k) :
            _Oa, _Ob, _max_dist = self.choose_distance_objects(self.o_pair, _dist_set)
            if (_max_dist == 0) :
                break
            else:
                self.calculate_projection_distance(self.o_pair, _dist_set, _Oa, _Ob)
                _col += 1
                self._col = _col
                if(_col < self.k):
                    _new_dist_set = self.projection_on_hyper_plane(self.o_pair, _dist_set, self.obj_k_d, _col-1) 
                    #Base on previous xi, xj data to compute new distance
                    _dist_set = _new_dist_set
                else:
                    break       
                        
        return self.obj_k_d
    
    def plot(self, label_set):
        plt.xlabel("x-axis") 
        plt.ylabel("y-axis") 

        for i in range(len(self.obj_k_d)):
            _x = self.obj_k_d[i][0]
            _y = self.obj_k_d[i][1]
            _label = label_set[i]
            plt.plot(_x, _y, 'b.', markersize=10)
            plt.annotate(
                _label, xy = (_x, _y), xytext = (30, 20), textcoords = 'offset points', ha = 'right', va = 'bottom', 
                bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.1),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        plt.show()

           
if __name__ == '__main__':
    '''
        Main program.
            For the FastMap class execution..

    '''
   
    object_pair, distance = getInputData('fastmap-data.txt')
    #print ('object_pair=', object_pair)   
    #print ('distance=' , distance)
     
    dimension = 2
    fm = FastMap()
    k_d_distance = fm.execute(object_pair, distance, dimension)
    
    print ('The ', dimension, ' dimensions result = \n', k_d_distance)
    
    label_set = getObjectName('fastmap-wordlist.txt')
    
    fm.plot(label_set)
