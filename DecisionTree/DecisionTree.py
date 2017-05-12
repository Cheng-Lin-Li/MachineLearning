#!/usr/bin/env python
# encoding: utf-8
'''
Machine Learning Algorithm Name: Decision Tree

This is a sample program to demonstrate the implementation of decision tree by Entropy.

@author: Cheng-Lin Li a.k.a. Clark

@copyright:    2017 Cheng-Lin Li@University of Southern California. All rights reserved.

@license:    Licensed under the GNU v3.0. https://www.gnu.org/licenses/gpl.html

@contact:    clark.cl.li@gmail.com
@version:    1.0

@create:    September, 11, 2016
@updated:    February, 2, 2017

'''

import sys
import math

__all__ = []
__version__ = 1.0
__date__ = '2016-09-11'
__updated__ = '2017-02-02'

DEBUG = 0
DELIMITER = ','
TRAINING_SET = []
ATTRIBUTE='attr_name'
DATA='data'
TREEROOT = 'root'
TREEEND = '.'
PRINTBRANCH = True
TESTING_SET = [{'attr_name': 'Size', 'data': ['Large']}, {'attr_name': 'Occupied', 'data': ['Moderate']}, {'attr_name': 'Price', 'data': ['Cheap']}, {'attr_name': 'Music', 'data': ['Loud']}, {'attr_name': 'Location', 'data': ['City-Center']}, {'attr_name': 'VIP', 'data': ['No']}, {'attr_name': 'Favorite Beer', 'data': ['No']}]
TESTING_SET2 = [{'attr_name': 'Size', 'data': ['Large']}, {'attr_name': 'Occupied', 'data': ['Moderate']}, {'attr_name': 'Price', 'data': ['Cheap']}, {'attr_name': 'Music', 'data': ['Loud']}, {'attr_name': 'Location', 'data': ['City-Center']}, {'attr_name': 'VIP', 'data': ['No']}, {'attr_name': 'Favorite Beer', 'data': ['Yes']}]
TESTING_SET3 = [{'attr_name': 'Size', 'data': ['Large']}, {'attr_name': 'Occupied', 'data': ['Moderate']}, {'attr_name': 'Price', 'data': ['Normal']}, {'attr_name': 'Music', 'data': ['Loud']}, {'attr_name': 'Location', 'data': ['City-Center']}, {'attr_name': 'VIP', 'data': ['No']}, {'attr_name': 'Favorite Beer', 'data': ['Yes']}]




# Can be improved to get testing data set from input file if resource is available.

def sanitize(data):
    if '(' in data:
        return data.strip().replace('(' ,'')
    elif ')' in data:
        return data.strip().replace(')', '')
    elif ':' in data:
        return data.split(':')[1].strip()
    elif ';' in data:
        return data.strip().replace (';', '')    
    else:
        return data.strip()

def getInputData(filename):
#
# Get data file and perform data format verification. 
# Can be improved: to get testing data set from input file if resource is available.
#  Leverage data structure: Dictionary(Map) to store data for each attribute.
#    - Attribute Dictionary{} include attribute_name, and attribute_data.
#  All training set data will be store in a List[] will contain for all attribute_data Dictionaries.
#  Training_Set= [Attribute1{'attr_name':'att1', 'data':['type1, type1, type2...']}, Attribute2{},...,AttributeN{}]
#
    i = 0
    training_data = []
    try:
        with open(filename, 'r') as _fp:
            for _each_line in _fp:
                if i == 0: #Attribute Name
                    attribute_name_list = [str(sanitize(each_name)) for each_name in _each_line.split(DELIMITER)]
                    if DEBUG > 0:  print ('attribute_name_list=' + str(attribute_name_list))
                    for each_attribute_name in attribute_name_list:
                        training_data.append({ATTRIBUTE:str(each_attribute_name), DATA:[]}) #Put each attribute dictionary into training set
                        if DEBUG > 1:  print ('TRAINING_SET= ' + str(TRAINING_SET))
                    i += 1
                elif i == 1: #Space
                    i += 1
                else: #Attribute Data
                    _attribute_data_list = [str(sanitize(each_data)) for each_data in _each_line.split(DELIMITER)]
                    if DEBUG > 1: print ('attribute_data_list' + str(attribute_data_list))
                    for index, each_data in enumerate(_attribute_data_list):
                        training_data[index][DATA].append(str(each_data))

            #Debug purpose code:
            if DEBUG > 0:  
                print ('TRAINING_SET= ')
                for each_attribute in training_data:
                    print (str(each_attribute))
        return training_data
    except IOError as _err:
        print ('File error: ' + str (_err))
        exit()
    
    
#
# Main Class: DecisionTree
#    DecisionTree Machine implementation.
#    
# Decision Tree Data Structure
# DecisionTree { 
#    node = ''   
#    branch = [] #Branch List
#    child_tree = [DecisionTree] #List of DecisionTree
# }
#

class DecisionTree (object):
#DecisionTress Class
# Can be improved: Another approach is inheriting from a tree class if resource is available.
#
    def __init__(self, training_set=[], testing_set=[], min_inf_gain=0.0):
        # initialization
        self.node = ''
        self.branchset = {} #{{'branch_name1':{'classification_type1': counter,..., 'classification_typeN': counter}, 'branch_name2':{...}}}
        self.child_tree = {} #{'branch_name1':DecisionTree1, 'branch_name2':DecisionTree2...}
        self.minInfoGain = min_inf_gain
        self.training_set = training_set
        self.testing_set = testing_set
        self.classification_name=''
        
        if len(training_set) >0 :
            self.takeTraining(training_set)
        else:
            pass
        
        if len(testing_set) > 0:
            self.predictTesting(testing_set)
        else:
            pass
        
    def setNode (self, best_attribute):
        self.node = str(best_attribute)
        
    def getNode (self):
        return self.node
        
    def setClassificationName(self, classification_attr):
        self.classification_name = classification_attr[ATTRIBUTE]
        
    def getClassificationName(self):
        return self.classification_name
                    
    def setMinInfoGain (self, minInfoG):
        self.minInfoGain = minInfoG   
         
    def getMinInfoGain (self):
        return self.minInfoGain
    
    def setBranchSet (self, branch_set):
        # Create branch data set for an node
        # Branch set will include individual branch type with name, classification name list with counter information.
        # 
        #    branch_set include: {{'branch_name1':{'classification_type1': counter,..., 'classification_typeN': counter}, 'branch_name2':{...}}} 
        self.branchset = branch_set
        
    def getBranchSet (self):
        return self.branchset
                
    def setChildTree (self, branch_type):
        #Create Child Tree base on best information gain attribute into child tree list
        #Or append new child tree into child tree dictionary
        #{'branch_name1':DecisionTree1, 'branch_name2':DecisionTree2...}
        dt = DecisionTree()
        self.child_tree.update({str(branch_type): dt})
        return dt
    
    def getChildTree(self):
        #Get child tree dictionary
        return self.child_tree    
        
    def individBranchEntropy (self, branch, branch_total):   
        #Calculate the entropy value of individual branch with specific attribute/node. 
        if len(branch) == 0:
            return 0
        else:
            _branch_info = 0
            for each_class_type in branch:
            #Get Entropy of each branch
                _class_type_count = branch[each_class_type]
                _branch_info += _class_type_count/branch_total*math.log2(branch_total/_class_type_count)
                # The other form of formula
                #_branch_info -= _class_type_count/branch_total*math.log2(_class_type_count/branch_total)
            return _branch_info
        
    def sumBranchEntropy (self, attribute, classification_attribute):
        #Calculate the sum of entropy value by each weighted branch.
        #    The "total" branch count will speed up calculation efficiency
        #
        #print ('sumBranchEntropy')
        branch_set = {}
        #    branch_set include: {{'branch_name1':{'classification_type1': counter,..., 'classification_typeN': counter}, 'branch_name2':{...}}}       
        
        branch_total_count={} 
        #    branch_total_count include: {'branch_name1':total_count_of_record,..., 'branch_nameN': total_count_of_record}
        
        class_attr = classification_attribute
        info = 0.0        
        n = len(classification_attribute[DATA])

        if n == 0:
            return 0
        else:
            for index, each_attr_type in enumerate(attribute[DATA]):
            #Count the number of each type in classification attribute
                if branch_set.get(each_attr_type):
                # if the branch type exist
                    if branch_set[each_attr_type].get(str(class_attr[DATA][index])):
                    # if the branch type + classification type exist
                        branch_set[each_attr_type][class_attr[DATA][index]] += 1
                        branch_total_count[each_attr_type] += 1
                        if DEBUG > 1: print ('3.1. branch_set=' + str(branch_set))
                    else:
                    # if the branch type exist but classification not exist, create it and set counter to 1
                        branch_set[each_attr_type][class_attr[DATA][index]] = 1
                        branch_total_count[each_attr_type] += 1
                else:
                # if the branch type not exist, create a dictionary and put classification type with counter set to 1
                    branch_set[each_attr_type] = {class_attr[DATA][index]:1}
                    branch_total_count[each_attr_type] = 1
                    
            for index, each_attr_type in enumerate(branch_set):
            # Calculate summary branch entropy
            #    For each branch (attribute type), calculate the entropy
            #    branch_set include: {{'branch_name1':{'classification_type1': counter,..., 'classification_typeN': counter}, 'branch_name2':{...}}}                       
            #    branch_total_count include: {'branch_name1':total_count_of_record,..., 'branch_nameN': total_count_of_record}
                _branch_total = branch_total_count[each_attr_type] #branch record count
                _weight = _branch_total/n #weighting of each branch
                _branch_info = 0           
                _branch_info = _weight * self.individBranchEntropy(branch_set[each_attr_type], _branch_total)
                info += _branch_info     
            info = -1 * info      
            return info, branch_set
        
    def getNodeEntropy (self, classification_attribute):
        #Calculate Node Entropy
        if DEBUG > 1:  print ('getNodeEntropy: Begin')
        counter_dict={}
        info = 0.0
        n = len(classification_attribute[DATA]) # total number of data in the attribute
        if n == 0:
            return 0
        else:
            for each_data in classification_attribute[DATA]:
            #Count the number of each type in classification attribute
                if counter_dict.get(each_data):
                    counter_dict[each_data] += 1
                else:
                    counter_dict[each_data] = 1    
            for each_type in counter_dict:
            # Calculate Node entropy
                info += (counter_dict[each_type]/n)*math.log2(n/counter_dict[each_type])

            if DEBUG > 1: print(', Node info =' + str(info))
            return info
            
    def getNodeInfoGain (self, attribute, classification_attribute):
        # Information Gain = Node entropy - Sum of weighted branch entropy
        if DEBUG > 1: print ('Get Information Gain')
        _nodeEntropy = self.getNodeEntropy(classification_attribute)
        _sumBranchEntropy, _branch_set = self.sumBranchEntropy(attribute, classification_attribute)
        return (_nodeEntropy - _sumBranchEntropy), _branch_set
    
    def NumClassification(self, training_set_classification):
        # Calculate the number of type in the specific classification. 
        # If all training data are same classification, return 0
        return len(set(training_set_classification))
    
    def getSplitTrainingSetByAttr(self, best_attribute, branch_set, training_set):     
        # Split training set into subset by best information gain attribute & attribute type. 
        # {attr_type1:Training_set1[],...,attr_typeN:Training_setN[]}
        # Training_Set= [Attribute1{'attr_name':'att1', 'data':['type1, type1, type2...']}, Attribute2{},...,AttributeN{}]
        # The training will exclude the best information gain attribute 
        if DEBUG > 1: print ('getSplitTrainingSetByAttr')
        _r_training_set = training_set
        split_training_dict = {}
        try:
            _r_training_set.remove(best_attribute) #Remove best information gain attribute from training set 
        except ValueError:
            print('ValueError: the best attribute not in the training set, cannot be removed.')
        for each_type in branch_set:
        # Example: each_type = (Large, Small, Medium)
        # Initialized training set for each attribute type. The result will be: split_training_dict={'Large': [], 'Medium': [], 'Small': []}
            split_training_dict[each_type]=[]
            for each_attr in training_set:
            #split_training_dict[each_type]=[{'attr_name':'Occupied', 'DATA':[]},...,{}]
            #each_attr={'data': ['High', 'High', 'Moderate',..., 'Low'], 'attr_name': 'Occupied'}
                split_training_dict[each_type].append({ATTRIBUTE:str(each_attr[ATTRIBUTE]), DATA:[]})
                #split_training_dict{'Medium': [{'attr_name': 'Occupied', 'data': []}, {'attr_name': 'Price', 'data': []}, {'attr_name': 'Music', 'data': []}, {'attr_name': 'Location', 'data': []}, {'attr_name': 'VIP', 'data': []}, {'attr_name': 'Favorite Beer', 'data': []}, {'attr_name': 'Enjoy', 'data': []}], 'Small': [{'attr_name': 'Occupied', 'data': []}, {'attr_name': 'Price', 'data': []}, {'attr_name': 'Music', 'data': []}, {'attr_name': 'Location', 'data': []}, {'attr_name': 'VIP', 'data': []}, {'attr_name': 'Favorite Beer', 'data': []}, {'attr_name': 'Enjoy', 'data': []}], 'Large': [{'attr_name': 'Occupied', 'data': []}, {'attr_name': 'Price', 'data': []}, {'attr_name': 'Music', 'data': []}, {'attr_name': 'Location', 'data': []}, {'attr_name': 'VIP', 'data': []}, {'attr_name': 'Favorite Beer', 'data': []}, {'attr_name': 'Enjoy', 'data': []}]}
                
        for index1, each_type in enumerate(best_attribute[DATA]):
        # Get training data for each attribute type in best information gain attribute 
        #    example: each_type in best attribute = Large, Medium, Small, Large, Large, Small,...
        
            for index2, each_attr in enumerate(_r_training_set):
            #  Add training data into new split training data set by each attribute type 
                split_training_dict[each_type][index2][DATA].append(str(each_attr[DATA][index1]))
                
                #Debug purpose code:
                if DEBUG > 1: print('each_attr[DATA][index1]=' + str(each_attr[DATA][index1]))
                if DEBUG > 1: print('==>split_training_dict[each_type][index2]=' + str(split_training_dict[each_type][index2]))  
        return split_training_dict
            
    def setTreeLeaf(self, class_attr):
        #Construct node of tree end  
        #    Set Node = final classification type Name, example, 'Yes' or 'No'
        #    Set branch set, example = {{'Yes':{'Yes':3}, {'No':{'No':1}}
        #     class_attr, example = {'data': ['Yes', 'Yes', 'No', 'No'], 'attr_name': 'Enjoy'}
        #     each_attr_type in class_attr[DATA], example = Yes or No
        if DEBUG > 1: print('Construct Tree Leaf: Begin')     
        branch_set = {}
        for each_attr_type in class_attr[DATA]:
            if branch_set.get(each_attr_type):
            # if the branch type exist
                branch_set[each_attr_type][each_attr_type] += 1 
            else:
                branch_set[each_attr_type] = {each_attr_type:1}
        _best_type = ""
        _best_count = 0
        for each_branch in branch_set :
            _count = branch_set[each_branch][each_branch]
            if _count > _best_count:
                _best_count = _count
                _best_type = each_branch
        self.setNode(_best_type)
        self.setBranchSet(branch_set)
        self.setChildTree(TREEEND)              
         
    def takeTraining (self, training_set, predict_attr_index=-1):
        # if predict_attr_index == -1 which means the last one of attribute in list is the classification target. 
        # default the last attribute in training list/set is classification attribute
        #      
        #print ('takeTraining: Begin')
        _mini_Info_gain = self.getMinInfoGain() #default 0
        _split_training_dict = {}               
        _classification_attr = training_set[predict_attr_index]  # List of dictionary to store classification and counts {'classification':'result', 'count':integer}
        info_gain = 0
        best_info_gain = 0
        best_attribute = dict()
        best_branch_set = []
        
        self.setClassificationName(_classification_attr)
        #  Training_Set= [Attribute1{'attr_name':'att1', 'data':['type1, type1, type2...']}, Attribute2{},...,AttributeN{}]

        for each_attribute in training_set:
        #Calculate Entropy for each attribute
            if each_attribute[ATTRIBUTE] != self.classification_name: #Exclude information gain calculation from classification attribute
                info_gain, branch_set = self.getNodeInfoGain(each_attribute, _classification_attr)                                               
                if info_gain > best_info_gain : # Assume the order of attribute is prioritized. the lower index has higher priority.
                #Select best attribute
                # if this attribute has a greater information gain then previous best attribute, we replace the best attribute to current one.    
                    best_info_gain = info_gain
                    best_attribute = each_attribute
                    best_branch_set = branch_set


                else :    
                    pass #Calculate for next Attribute
            else:
                pass # Escape classification attribute and process next attribute

            if (best_info_gain <= _mini_Info_gain) or (self.NumClassification(_classification_attr)==1) or (len(training_set) == 1) :
            #    Termination condition for leaf node
            #        1. Information Gain <= Minimum Information Gain we expect, default = 0
            #        2. All classification data are same type
            #        3. No more attribute have to be processed.
            #    This is tree end situation
                if DEBUG > 1: print("****Termination Condition****")
                self.setTreeLeaf(_classification_attr)
                return    
                         
        if best_attribute : 
            self.setNode(best_attribute[ATTRIBUTE])
            self.setBranchSet(best_branch_set) 
            _split_training_dict = self.getSplitTrainingSetByAttr(best_attribute, best_branch_set, training_set)
            # Split training data by each branch
            _class_name = self.getClassificationName()
        
            for each_branch in best_branch_set:
            # Recursive for each branch as subtree to take training
            # each_branch = (Large, Small, Medium) or Yes, No
            #  
                _pd_idx = 0
                for idx, each_att in enumerate(_split_training_dict[each_branch]):
                    #Get data index of classification attribute
                    if _class_name == str(each_att[ATTRIBUTE]):
                        _pd_att_idx = idx
                    else :
                        pass
                _dt = self.setChildTree(each_branch) 
                _dt.takeTraining(_split_training_dict[each_branch], _pd_att_idx) # Track point
        else:
            self.setNode('')

    def printDecisionTree (self, fg_branch = False):
        #Print the Decision Tree
        #    level-order traversal approach
        if DEBUG > 1: print ('Print Decision Tree: ')
        level = 0
        _dt = self
        _print_queue = [{TREEROOT:_dt}] # to store tree node for next level
        each_dict = {}

        while ( len(_print_queue) > 0 ):
            _pl = []
            _pl = list(_print_queue)
            _pl = _print_queue
            _print_queue = []
            each_dict.clear()
            while ( len(_pl) > 0 ) :     
                if fg_branch  :
                    for each_dict in _pl:
                        for each_data in each_dict:
                            #print branch
                            print(each_data + ',', end='') 
                    print()  
                  
                for each_dict in _pl:                                  
                    for each_data in each_dict:
                        _dt = each_dict[each_data]
                        print(_dt.getNode() + ', ', end='')
                        _print_queue.append(dict(_dt.getChildTree()))                                       
                _pl = []
            print()
            level+= 1                           
        return   

    def predictTesting (self, dt, testing_set, level=0):
        _dt = dt
        _sub_tree_dict = _dt.getChildTree()
        _test_attr_branch= ''
        
        if bool(_sub_tree_dict) :    
            _attr = _dt.getNode()
            for each_att in testing_set:
            #each_att={'data': ['Large'], 'attr_name': 'Size'}
                if each_att[ATTRIBUTE]==_attr :
                    _test_attr_branch=each_att[DATA][0]
                    ###################
                    print('\tLevel '+ str(level) +': '+ each_att[ATTRIBUTE]+'='+str(_test_attr_branch))
                    break          
            if _test_attr_branch in _sub_tree_dict:             
                _dt = _sub_tree_dict[_test_attr_branch]  
                if len(_dt.getChildTree()) == 1 and TREEEND in _dt.getChildTree() :
                # This is tree end.
                    return _dt.getNode() 
                else :             
                    _result = self.predictTesting(_dt, testing_set, level+1)
            else:
                _result = 'NaN-Can not predict: "'+each_att[ATTRIBUTE]+' = '+ str(_test_attr_branch) + '" not in tree node.\n      ==>Tree Node Data=('+str(_sub_tree_dict)+')' 
            return _result
        else:
        # Tree End 
            return _dt.getNode()
          
    def getTestResult(self, testing_set):
        print()
        print('Testing:===========')
        print('Testing Data=' + str(testing_set))
        print()      
        _rt = self.predictTesting(self, testing_set)
        print()          
        print('Result==>'+ str(self.classification_name) +'='+ str(_rt)+'\n\n')
    


if __name__ == "__main__":

    '''
        Main program.
            Construct Decision tree with training data.
            Print Decision Tree model after training
            Predict test data and print result
    '''
    program_name = sys.argv[0]
    input_file = ''
    if len(sys.argv) < 2: 
        print ('Input data file missing!! Please put dt-data.txt into the same folder.')
    else:
        input_file = sys.argv[1]
        
    TRAINING_SET = getInputData(input_file)
    
    dt = DecisionTree()
    
    dt.takeTraining(TRAINING_SET)
    
    dt.printDecisionTree()
    dt.getTestResult(TESTING_SET)
    print ('==== Decision Tree ====')
    dt.printDecisionTree(PRINTBRANCH)
    print ('Additional Testing Data:', end='\n\n')
    dt.getTestResult(TESTING_SET2)
    dt.getTestResult(TESTING_SET3)
