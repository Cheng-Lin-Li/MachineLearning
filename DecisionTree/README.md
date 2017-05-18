## This is an implementation of Decision Tree Algorithm in Python 3

## The task:

This exercise is to predict whether you will have a good night-out in Jerusalem for the coming New Year's Eve. 
Assume that you have kept a record of your previous night-outs with the following attributes. 

	• The size of the place {Large, Medium, Small}
	
	• How densely the place is usually occupied {High, Moderate, Low} 
	
	• How the prices are {Expensive, Normal, Cheap} 
	
	• Volume of the music {Loud, Quiet} 
	
	• The location {Talpiot, City-Center, Mahane-Yehuda, Ein-Karem, German-Colony}
	
	• Whether you are a frequent customer (VIP) {Yes, No} 
	
	• Whether this place has your favorite beer {Yes, No} 
	
	• Whether you enjoyed {Yes, No}

#### Usage: python DecisionTree.py dt-data.txt	

#### Input: A data file (dt-data.txt) that contains the relevant records.

in case there is a tie between attributes, the program use the order that the attributes are listed as the priority. 
For example, to break tie between prices and location, since prices are listed closer to the front of the list, use prices instead of location. 

The implementation writes a program to construct decision trees based on the idea of splitting by Information Gain. 
Make a prediction for (size = Large; occupied = Moderate; price = Cheap; music = Loud; location = City-Center; VIP = No; favorite beer = No).


The prediction request is assigned into global variable in the program as below.

TESTING_SET = [{'attr_name': 'Size', 'data': ['Large']}, {'attr_name': 'Occupied', 'data': ['Moderate']}, {'attr_name': 'Price', 'data': ['Cheap']}, {'attr_name': 'Music', 'data': ['Loud']}, {'attr_name': 'Location', 'data': ['City-Center']}, {'attr_name': 'VIP', 'data': ['No']}, {'attr_name': 'Favorite Beer', 'data': ['No']}]

There are two additional test cases for reference.


TESTING_SET2 = [{'attr_name': 'Size', 'data': ['Large']}, {'attr_name': 'Occupied', 'data': ['Moderate']}, {'attr_name': 'Price', 'data': ['Cheap']}, {'attr_name': 'Music', 'data': ['Loud']}, {'attr_name': 'Location', 'data': ['City-Center']}, {'attr_name': 'VIP', 'data': ['No']}, {'attr_name': 'Favorite Beer', 'data': ['Yes']}]


TESTING_SET3 = [{'attr_name': 'Size', 'data': ['Large']}, {'attr_name': 'Occupied', 'data': ['Moderate']}, {'attr_name': 'Price', 'data': ['Normal']}, {'attr_name': 'Music', 'data': ['Loud']}, {'attr_name': 'Location', 'data': ['City-Center']}, {'attr_name': 'VIP', 'data': ['No']}, {'attr_name': 'Favorite Beer', 'data': ['Yes']}]

It is easy task to modify the code and get those parameters dynamically load during program execution time.

The code prints the decision tree that it produces, the format is:
attribute on the 1st level
1st attribute on the 2nd level, 2nd attribute on the 2nd level, 3rd attribute on the 2nd level ...

For example, if the program uses 'size' as the root, occupied, prices and music as the attributes on the second level corresponding to size={Large, Medium, Small} respectively, then it should look like the following:

#### Output:

size 

occupied, prices, music

… More levels and attributes … 

(The final level)Yes, No, Yes, Yes, …


## Data Structure

	TRAINING_SET = [{'attr_name':'attribute1', 'data':['type1, type1, type2...']}, {}, ..., {}]  //Store the training data. e.g. { 'attr_name': 'Occupied', 'data': ['High', 'High', 'Moderate',..., 'Low']}

	TESTING_SET = [{'attr_name':'attribute1', 'data':['type1, type1, type2...']}, {}, ..., {}]  //Store the testing data

	branchset = {{'branch_name1':{'classification_type1': counter,..., 'classification_typeN': counter}, 'branch_name2':{...}}}  //Store the branches of each node

	child_tree = {{'branch_name1':DecisionTree1, 'branch_name2':DecisionTree2 ...}}  //Store the child trees of each node

	branch_total_count = {{'branch_name1':total_count_of_record,..., 'branch_nameN': total_count_of_record}}  //Store the number of cases in each branch of the node


## Main Methods

	def __init__  //Initialize decision tree

	def setBranchSet  //Create branch data set for an node

	def setChildTree  //Create Child Tree base on best information gain attribute into child tree list

	def individBranchEntropy  //Calculate the entropy value of individual branch with specific attribute/node. 

	def sumBranchEntropy  //Calculate the sum of entropy value by each weighted branch.

	def getNodeEntropy  //Caluculate Node Entropy

	def getNodeInfoGain  //Information Gain = Node entropy - Sum of weighted branch entropy

	def NumClassfication   //Calculate the number of type in the specific classification

	def getSplitTrainingSetByAttr  //Split training set into subset by best information gain attribute & attribute type. 

	def setTreeLeaf  //Construct node of tree end

	def takeTraining  //Default the last attribute in training list/set is classification attribute

	def printDecisionTree  //Print the decision tree

	def printNextLevelDecisionTree  //Visit each node in same level with tree end condition


## Process

  1. Create the data structure for training dataset and decision tree

  2. Initialize the decision tree

  3. Calculate the best information gain and select the best feature to split

  4. Create the child decision tree recursively and select the best feature for each child tree.

  5. Calculate the entropy and record the label during building tree

  6. Return the tree

  7. Make prediction for the testing data


