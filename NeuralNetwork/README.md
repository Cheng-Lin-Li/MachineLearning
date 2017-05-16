## This is an implementation of Neural Network Algorithm in Python 3

## Machine Learning Algorithm: Neural Network Algorithm.

## Implementation features:
The neural network support multiple layers with multiple dimensions input and one dimension output.

## Task:
In the directory gestures, there is a set of images that display "down" gestures (i.e., thumbs-down images) or other gestures. In this assignment, the program is required to implement the Back Propagation algorithm for Feed Forward Neural Networks to learn down gestures from training images available in downgesture_train.list. 


The label of an image is 1 if the word "down" is in its file name; otherwise the label is 0. The pixels of an image use the gray scale ranging from 0 to 1. In your network, use one input layer, one hidden layer of size 100, and one output node. Use the value 0.1 for the learning rate. For each perceptron, use the sigmoid function Ɵ(s) = 1/(1+e^-s). Use 1000 training epochs; initialize all w randomly between 0 to 1; and then use the trained network to predict the labels for the gestures in the test images available in downgesture_test.list. For the error function, use the standard least square error function. Output predictions and accuracy.


The image file format is "pgm" <http://netpbm.sourceforge.net/doc/pgm.html>. You may follow the link for the format details. 


##Objectives
Implement the Back Propagation algorithm for Feed Forward Neural Networks to learn down gestures from training images available in downgesture_train.list. After the training, use the trained network to predict the labels for the gestures in the test images available in downgesture_test.list.

The result should print out whether the prediction is correct for “down” gesture, the prediction result and the accuracy of prediction.

There are 184 training data sets in downgesture_train.list and 83 testing data sets in downgesture_test.list. The training data sets and testing data sets are PGM image, which is a special to present grayscale graphic image.

#### Usage: python NeuralNetwork.py

#### Input: A data file (downgesture_train.list) that contains 184 training data sets. A data file (downgesture_test.list) that contains 83 testing data sets. 
Both file names and rest of neural network parameters defined in the global variable section. It is easy task to modify the code and get those parameters dynamically load during program execution time.

Major parameters show as below:

LEARNING_RATE = 0.1 # Default learning rate
ITERATION = 1000
HIDDEN_LAYER_SIZES = [100,] #Hidden layer structure. The definition of [100, 10] is for multiple hidden layers, first layer with 100 neurals, and second hidden layer with 10 neurals,  
LOWER_BOUND_INIT_WEIGHT = 0 #Lower bound of weight for each connection.
UPPER_BOUND_INIT_WEIGHT = 1 #Upper bound of weight for each connection.
BINARY_CLASSIFICATION = True #The output will be either 0 or 1 if it is True or present the actual output value if it set to False.
THRESHOLD = 0.5 # Threshold for logistic activation function.
TOLERANCE = 1e-6 # Threshold of output delta for neural network converge.
CONSECUTIVE_TIMES = 10 # How many consecutive times the output delta less than tolerance to stop the training.

## Data Structure
Training data and testing data are stored in array like [[data1], [data2], ... , [dataN]]. For [data1] = [value_dimension_1, value_dimension_2, ..., value_dimension_d].


Labels of each data set are stored in list.


Weights store in an array like [[Weights of level-01], [Weights of level-12], ..., [Weights of level-(L-1)(L)]]. For [Weights of Level-01]=[[w01, w02, ..., w0d], [w11, w12, ..., w1d], ... [wd1, wd2, ..., wdd]]


The layer sizes are store in an array, in which each item corresponds to different layer.


Weight vector store in an array/vector.


## Main Methods
A Neural Network class supports on both logistic sigmoid and hyperbolic tangent functions as activation functions, resolver is stochastic gradient descent to implement and provide a lot of value sets for reference.


1. This implementation apply Stochastic learning approach where single example is randomly chosen from training set at each iteration.


2. The initial data range of weight matrix/array can be configured from parameters.


3. The output is one dimension.
        


self.hidden_layer_sizes = [hidden_layer1_sizes, hidden_layer2_sizes, ... , hidden_layerL-1_sizes]     


self.weights = weights array = [[Weights of level-01], [Weights of level-12], ..., [Weights of level-(L-1)(L)]]. 


     For [Weights of Level-01]=[[w01, w02, ..., w0d], [w11, w12, ..., w1d], ... [wd1, wd2, ..., wdd]]


self.max_iteration = integer, max. iterations;


self.learning_rate = float.


self.training_data = input array = [[data1], [data2], ... , [dataN]]. For [data1] = [value_dimension_1, value_dimension_2, ..., value_dimension_d]


self.input_numbers= integer, default is 0. Numbers of data set,


self.input_dimensions =integer, default is 0. Dimensions of data set, 


self.training_data_label = np.array(training_data_label)


self.output_numbers= integer, default is 0. Out put label numbers.


self.output_dimensions integer, default is 0. The dimensions will match with label data set automatically. 



self.network_layer_sizes = [input layer sizes+1, hidden layer1 sizes, ... , hidden layerL-1 sizes, output layer sizes ] 


    input layer sizes = input data dimensions., output layer sizes = label data dimensions.


self.tol = float. Tolerance of output and label data. If the tolerance between output & label data less than tol parameter by consecutive times, program will stop. 


## Process
Get input training data sets from files.


Assign label to each data set according to its file name. If ‘down’ exists in file name, the label is 1, otherwise 0.


Class constructor initialize the hidden layers array, network layers array, activation function and other parameters according to user’s input.


Execute algorithm by execute() function.


Initialize weights vector randomly with each value between given range.


A while-loop trains the neural network with all training data sets and initial weight vector.


According to Stochastic Learning approach, randomly pick up a data set from all training data sets.


Compute $$X_j^((l) )=θ(∑_(i=0)^(d(l-1))▒〖w_ij^((l)) X_i^((l-1)) 〗)$$ in the forward direction data set chosen in a. θ is an activation function, which can be both logistic sigmoid and hyperbolic tangent functions as what we set.	


Compute $$δ_j^((l-1))=(1-(X_i^((l-1)) )^2 ) ∑_(j=1)^(d(l))▒W_ij^((l))  δ_j^((l))$$ in the backward direction with corresponding result in step b.


Update weight vectors w_ij^((l)) by w_ij^((l))= w_ij^((l))- 〖η*X〗_i^((l-1) )*δ_j^((l)) . η is learning rate.


The program terminates when the maximum iteration was reached. The default iteration here is 1000.


Get input testing data sets from files.


Predict the label of each data set with trained neural network. If the prediction finds out the “down” gesture correctly, print “Match(O)”, otherwise “Match(X)”. Calculate and output the accuracy finally.

