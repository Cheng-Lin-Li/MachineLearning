#!/usr/bin/env python
# encoding: utf-8
'''
Machine Learning Algorithm Name: Neural Network

This is a sample program to demonstrate the implementation of Multilayer Neural Network.

@author: Cheng-Lin Li a.k.a. Clark

@copyright:  2017 Cheng-Lin Li@University of Southern California. All rights reserved.

@license:    Licensed under the GNU v3.0. https://www.gnu.org/licenses/gpl.html

@contact:    jianfali@usc.edu, clark.cl.li@gmail.com
@version:    2.1

@create:    October 28, 2016
@updated:   May, 15, 2017

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
'''

import numpy as np

TRAINING_SET = 'downgesture_train.list'
TESTING_SET = 'downgesture_test.list'
DEFAULT_ACTIVATION = 'logistic'
LEARNING_RATE = 0.1 # Default learning rate
ITERATION = 1000
HIDDEN_LAYER_SIZES = [100,] #Hidden layer structure. The definition of [100, 10] is for multiple hidden layers, first layer with 100 neurals, and second hidden layer with 10 neurals,  
LOWER_BOUND_INIT_WEIGHT = 0 #Lower bound of weight for each connection.
UPPER_BOUND_INIT_WEIGHT = 1 #Upper bound of weight for each connection.
BINARY_CLASSIFICATION = True #The output will be either 0 or 1 if it is True or present the actual output value if it set to False.
THRESHOLD = 0.5 # Threshold for logistic activation function.
TOLERANCE = 1e-6 # Threshold of output delta for neural network converge.
CONSECUTIVE_TIMES = 10 # How many consecutive times the output delta less than tolerance to stop the training.

def load_pgm_image(pgm):
    with open(pgm, 'rb') as f:
        f.readline()   # skip P5
        f.readline()   # skip the comment line
        xs, ys = f.readline().split()  # size of the image
        xs = int(xs)
        ys = int(ys)
        max_scale = int(f.readline().strip())

        image = []
        for _ in range(xs * ys):
            image.append(f.read(1)[0] / max_scale)

        return image



class NeuralNetwork(object):
    '''
    classdocs
    
    A NeuralNetwork class is implemented and provide a lot of value sets for reference.

    function lists:
    0. __init__(self, hidden_layer_sizes=HIDDEN_LAYER_SIZES, activation=DEFAULT_ACTIVATION, iteration=ITERATION, learning_rate=LEARNING_RATE, training_data=None, training_data_label=None, weight_low=0, weight_high=1, enable_binary_classification=True):
        NeuralNetwork class constructor.
        weight matrix initialization range will be [weight_low, weight_high), default is [0,1)
        enable_binary_classification will transform the output to binary classification if the value set to True. Default is True.
    1. logistic(self, x):
        Set logistic function as activation function.
    2. logistic_derivation(self, logistic_x):
        The derivation of logistic function. The input is the result of logistic(x).
    3. tanh(self, x):
        Set hyperbolic tangent function as activation function.
    4. tanh_derivation(self,tanh_x):
        The derivation of hyperbolic tangent function. The input is the result of tanh(x)
    5. initial_weights(self, network_layer_sizes):
        Set up the whole network layer. layer_sizes = [input_dimensions, hidden_layer1_sizes, ..., output_layer_sizes].
    6. set_layer_sizes(self, training_data, training_data_label):
        Construct the whole neural network structure, include [input layer sizes, hidden layer 1 sizes, ...hidden layer L sizes, output layer sizes]
        The input number of nodes/dimension and output number of nodes / dimensions will automatically define by training_data and training_data_label respectively.
    7. feed_forward(self, input_data):
        Neural Network feed forward propagation. It will return the calculation result of last/output layer which support multiple dimension.
        The output dimension will automatically adjust the dimension to fit with the dimensions of training data label.
    8. back_propagate(self, output, label_data):
        According to output data to perform back propagation on delta and weight array/matrix.
    9. predict(self, x):
        Return predict array to support multiple dimension results. The function also support output data transform to binary classification if the feature sets to True.
    

    '''   
    def __init__(self, hidden_layer_sizes=HIDDEN_LAYER_SIZES, activation=DEFAULT_ACTIVATION, iteration=ITERATION, learning_rate=LEARNING_RATE, 
                 training_data=None, training_data_label=None, weight_low=LOWER_BOUND_INIT_WEIGHT, weight_high=UPPER_BOUND_INIT_WEIGHT, enable_binary_classification = BINARY_CLASSIFICATION, tol = TOLERANCE):
        '''
        Constructor
        '''

        self.hidden_layer_sizes = np.array(hidden_layer_sizes)
                
        self.weights = np.array
        self.max_iteration = iteration
        self.learning_rate = learning_rate
        self.training_data = np.array(training_data)
        self.input_numbers = 0
        self.input_dimensions = 0
        self.training_data_label = np.array(training_data_label) 
        self.out_numbers = 0
        self.output_dimensions = 0
        self.X = []
        self.weight_low = weight_low
        self.weight_high = weight_high
        self.enable_binary_classification = enable_binary_classification
        self.tol = tol
        
        self.network_layer_sizes = np.array
        
        if activation == 'logistic':
            self.activation = self.logistic
            self.activation_derivation = self.logistic_derivation
        elif activation == 'tanh':
            self.activation = self.tanh
            self.activation_derivation = self.tanh_derivation
        else :
            pass

        if (self.training_data.ndim != 0 and self.training_data_label.ndim !=0):                                    
            self.execute (training_data, training_data_label)
        else:
            pass        
        
    def logistic(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def logistic_derivation(self, logistic_x):
        return logistic_x * (1.0-logistic_x)
   
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivation(self, tanh_x):
        return 1.0 - tanh_x**2
    
    def error_term_derivation(self, x, y):
        return 2 * (x-y)
        
    def initial_weights(self, network_layer_sizes):
        # network_layer_sizes = [input_dimensions, hidden_layer1_sizes, ..., output_layer_sizes].
        # range of weight values (-1,1)
        # self.weights = weights array = [[Weights of level-01], [Weights of level-12], ..., [Weights of level-(L-1)(L)]]. 
        # For [Weights of level-01]=[[w01, w02, ..., w0d], [w11, w12, ..., w1d], ... [wd1, wd2, ..., wdd]]
        
        weights = []
        scale = 0 # optimal scale of weight is m**(-1/2)
        for l in range(1, len(network_layer_sizes)):
                scale = (network_layer_sizes[l-1])**(-1/2)
                weights.append(((self.weight_high)-(self.weight_low))*np.random.normal(size=(network_layer_sizes[l-1], network_layer_sizes[l]))+(self.weight_low))                   
                #There are different approach to optimal the distribution of weights.
#                weights.append(((self.weight_high)-(self.weight_low))*np.random.normal(scale=_scale, size=(network_layer_sizes[l-1], network_layer_sizes[l]))+(self.weight_low))   
                np.random.random
        self.weights = weights
        return self.weights

    def set_layer_sizes(self, training_data, training_data_label):
        #Construct the whole neural network structure, include [input layer sizes, hidden layer 1 sizes, ...hidden layer L sizes, output layer sizes]
        network_layer_sizes = []
        dim = training_data.ndim;
        if dim != 0:
            self.input_numbers, self.input_dimensions = training_data.shape
        else:
            pass
        dim = training_data_label.ndim;
        if dim !=0:
            if dim == 1:
                self.output_numbers = training_data_label.shape[0]
                self.output_dimensions = 1;
            else:
                self.output_numbers, self.output_dimensions = training_data_label.shape
        else:
            pass
        
        network_layer_sizes.append(self.input_dimensions+1) # add X0
        
        for i in self.hidden_layer_sizes:
            network_layer_sizes.append(i)
        
        network_layer_sizes.append(self.output_dimensions) 
        self.network_layer_sizes = np.array(network_layer_sizes)

        return self.network_layer_sizes
    
    def feed_forward(self, input_data):
        X = [np.concatenate((np.ones(1).T, np.array(input_data)), axis=0)] #add bias unit [array([])]
        W = self.weights
        _wijxi = []
        _xj = []
        
        for l in range(0, len(W)):
            _wijxi = np.dot(X[l], W[l])
            _xj = self.activation(_wijxi)
            # Setup bias term for each hidden layer, x0=1
            if l < len(W)-1:
                _xj[0] = 1 
            X.append(_xj)  
            
        self.X = X
        return X[-1] #return the feed forward result of final level.         
    
    def back_propagate(self, output, label_data):
        X = self.X
        W = list(self.weights) #self.weights=<class list>[array([ndarray[100],ndarray[100],...X961]), array(ndarray[1],ndarray[1],...X100)]
        avg_err = []
        Delta = []
        _x = []
        _d = []
        _w = []
        _y = []
        
        _y = np.atleast_2d(label_data)   
        _x = np.atleast_2d(output)
        # Base level L delta calculation.
        avg_err = np.average(_x - _y)
        Delta = [self.error_term_derivation(_x, _y) * self.activation_derivation(_x)] # Delta = error term derivation * activation function derivation
        # #<class list>[array([])]
        
        # Calculate all deltas and adjust weights
        for l in range(len(X)-2, 0, -1):
            _d = np.atleast_2d(Delta[-1])
            _x = np.atleast_2d(X[l])
            _w = np.array(W[l])

            Delta.append( self.activation_derivation(_x) * Delta[-1].dot(_w.T) )    
            W[l] -= self.learning_rate * _x.T.dot(_d)

        #Calculate the weight of input layer and update weight array
        _x = np.atleast_2d(X[l-1])
        _d = np.atleast_2d(Delta[-1])            
        W[l-1] -= self.learning_rate * _x.T.dot(_d)        
        
        self.weights = W
        return avg_err
    
    def predict(self, x):
        _r = []
        _r = self.feed_forward(x[0])
        enable_binary_classification = self.enable_binary_classification
        
        # Enable the binary classification on predict results.
        if enable_binary_classification and self.activation == self.logistic:
            for i in range(len(_r)):
                if _r[i] >= THRESHOLD:
                    _r[i] = 1
                else:
                    _r[i] = 0
        else:
            pass
        return _r
    
    def execute(self, training_data, training_data_label):
        '''
        Execute function to train the neural network.
        '''
        self.training_data = np.array(training_data)
        self.training_data_label = np.array(training_data_label)
        network_layer_sizes = self.set_layer_sizes(self.training_data, self.training_data_label)        
        max_iter = self.max_iteration
        input_numbers = self.input_numbers
        avg_err = 0
        counter = 0
        
        self.initial_weights(network_layer_sizes)
        
        # Execute training.
        for idx in range (0, max_iter): 
            i = np.random.randint(self.training_data.shape[0])
            result = self.feed_forward(training_data[i])
            avg_err = self.back_propagate(result, training_data_label[i])
            if abs(avg_err) <= self.tol :
                counter += 1
                if counter >= CONSECUTIVE_TIMES:
                    break
                else:
                    pass
            else:
                _counter = 0
        print('Neural Network Converge at iteration =', idx+1)                
        print('Total input numbers=', input_numbers)    

'''
Main program for the NeuralNetwork class execution.

'''
           
if __name__ == '__main__':
    '''
        Main program.
            Read the labeled data from down gesture training list as training data.
            Construct Neural Network with training data.
            Read the test data from downgesture_test.list
            Calculate the accuracy of predictions.
    '''        

    images = []
    labels = []
    
    with open(TRAINING_SET) as f:
        for training_image in f.readlines():
            training_image = training_image.strip()
            images.append(load_pgm_image(training_image))
            if 'down' in training_image:
                labels.append([1,])
            else:
                labels.append([0,])  
    
    nn = NeuralNetwork(hidden_layer_sizes=HIDDEN_LAYER_SIZES, activation=DEFAULT_ACTIVATION, learning_rate=LEARNING_RATE, iteration=ITERATION, weight_low=LOWER_BOUND_INIT_WEIGHT, weight_high=UPPER_BOUND_INIT_WEIGHT, enable_binary_classification=BINARY_CLASSIFICATION)
    nn.execute(images, labels)
    total = 0
    correct = 0
    dim = np.array(labels).ndim;
    if dim == 1:
        threshold_array = np.array(THRESHOLD)
    else:
        threshold_array = np.array(THRESHOLD)*np.array(labels).shape[1]
    
    with open(TESTING_SET) as f:

        for test_image in f.readlines():
            total += 1
            test_image = test_image.strip()
            p = nn.predict([load_pgm_image(test_image),])

            if np.all(p >= threshold_array) == ('down' in test_image):
                if np.all(p >= threshold_array) :
                    print('Match(O)=>{}: Predict=True, Output value={}'.format(test_image, p)) 
                else:
                    print('Match(O)=>{}: Predict=False, Output value={}'.format(test_image, p))                   
                correct += 1
            else :
                if np.all(p >= threshold_array):
                    print('Match(X)=>{}: Predict=True, Output value={}'.format(test_image, p)) 
                else:
                    print('Match(X)=>{}: Predict=False, Output value={}'.format(test_image, p)) 
    #print(nn.weights)
    print('Accuracy: correct rate: {}%'.format(correct / total*100))
    