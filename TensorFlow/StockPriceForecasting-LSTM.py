'''
Project Name: Multi-Layer Perceptron (MLP), and Long-Short Term Memory (LSTMs) for stock price forecasting 

This is a sample program to demonstrate the implementation that leverage Tensorflow to construct a LSTMs for stock price forecasting.
Environments:
    1. Tensorflow version 1.1
    2. Python version 3.6
@author:    Cheng-Lin Li a.k.a. Clark

@copyright:  2017 Cheng-Lin Li@University of Southern California. All rights reserved.

@license:    Licensed under the GNU v3.0. https://www.gnu.org/licenses/gpl.html

@contact:    clark.cl.li@gmail.com
@version:    1.8

@create:    November, 29, 2016
@updated:   May, 17, 2017
'''
# # Stock Price Forecasting by Long-Short Term Memory (LSTMs) 
# This project will leverage the power of Long-Short Term Memory (LSTMs) on Stock Price prediction.
# 
# Multi-Layer Perceptron (MLP) will be a based line of performance as the benchmark.
# 

# ### Dataset Description:
# #### 1. Title
# 
# : Weekly stock data for Dow Jones Index
# 
# #### 2. Source:
# 
#  This dataset comprises data reported by the major stock exchanges.
# 
# #### 3. Past Usage
# This dataset was first used in:
# 
# Brown, M. S., Pelosi, M. & Dirska, H. (2013). Dynamic-radius Species-conserving Genetic Algorithm for 
# the Financial Forecasting of Dow Jones Index Stocks. Machine Learning and Data Mining in Pattern 
# Recognition, 7988, 27-41.
# 
# #### 4. Relevant Information
#     In predicting stock prices you collect data over some period of time - day, week, month, etc. But you cannot take advantage of data from a time period until the next increment of the time period. For example, assume you collect data daily.  When Monday is over you have all of the data for that day.  However you can invest on Monday, because you don't get the data until the    end of the day.  You can use the data from Monday to invest on Tuesday.  
# 
#     In our research each record (row) is data for a week.  Each record also has the percentage of return that stock has in the following week (percent_change_next_weeks_price). Ideally, you want to determine which stock will produce the greatest rate of return in the following week. This can help you train and test your algorithm.
# 
#     Training data vs Test data:
#     In (Brown, Pelosi & Dirska, 2013) we used quarter 1 (Jan-Mar) data for training and    quarter 2 (Apr-Jun) data for testing.
# 
#     Interesting data points:
#     If you use quarter 2 data for testing, you will notice something interesting in    the week ending 5/27/2011 every Dow Jones Index stock lost money.
# 
#     The Dow Jones Index stocks change over time. We indexed them as below sequence. The stocks that made up the index in 2011 were:
#         0. 3M             MMM
#         1. American Express     AXP
#         2. Alcoa            AA
#         3. AT&T             T
#         4. Bank of America        BAC
#         5. Boeing              BA
#         6. Caterpillar          CAT
#         7. Chevron          CVX
#         8. Cisco Systems         CSCO
#         9. Coca-Cola          KO
#         10. DuPont              DD
#         11. ExxonMobil          XOM
#         12. General Electric     GE
#         13. Hewlett-Packard        HPQ
#         14. The Home Depot          HD
#         15. Intel              INTC
#         16. IBM              IBM
#         17. Johnson & Johnson     JNJ    
#         18. JPMorgan Chase          JPM
#         19. Kraft            KRFT
#         20. McDonald's         MCD
#         21. Merck              MRK
#         22. Microsoft          MSFT
#         23. Pfizer              PFE
#         24. Procter & Gamble     PG
#         25. Travelers          TRV
#         26. United Technologies     UTX
#         27. Verizon          VZ
#         28. Wal-Mart          WMT
#         29. Walt Disney          DIS
# 
# #### 5. Number of Instances
# 
# There are 750 data records.  360 are from the first quarter of the year (Jan to Mar).
# 390 are from the second quarter of the year (Apr to Jun).
# 
# #### 6. Number of Attributes
# 
# There are 16 attributes.  
# 
# #### 7. For each Attribute
# 
#     a. quarter:  the yearly quarter (1 = Jan-Mar; 2 = Apr=Jun).
#     b. stock: the stock symbol (see above)
#     c. date: the last business day of the work (this is typically a Friday)
#     d. open: the price of the stock at the beginning of the week
#     e. high: the highest price of the stock during the week
#     f. low: the lowest price of the stock during the week
#     g. close: the price of the stock at the end of the week
#     h. volume: the number of shares of stock that traded hands in the week
#     i. percent_change_price: the percentage change in price throughout the week
#     j. percent_chagne_volume_over_last_wek: the percentage change in the number of shares of stock that traded hands for this week compared to the previous week
#     k. previous_weeks_volume: the number of shares of stock that traded hands in the previous week
#     l. days_to_next_dividend: the number of days until the next dividend
#     m. percent_return_next_dividend: the percentage of return on the next dividend 
#     n. ISM: ISM Manufacturing Index is based on surveys of more than 300 manufacturing firms by the Institute of Supply Management. The ISM Manufacturing Index monitors employment, production, inventories, new orders and supplier deliveries. A composite diffusion index monitors conditions in national manufacturing and is based on the data from these surveys. 
#     o. PCE: Personal Consumption Expenditure Price Index. It release every month.  
# 
# ###### Below data is for verification label.
# 
#     p. next_weeks_open: the opening price of the stock in the following week
#     q. next_weeks_close: the closing price of the stock in the following week
#     r. percent_change_next_weeks_price: the percentage change in price of the stock in the following week
# 

# ### Below program will read dataset and label into matrix.

# In[1]:


# Support for Python 2.7
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import sys

# Parameters for input data
FILE_NAME = 'dow_jones_index.csv'
RECORD_NUMBER = 15
STOCK_NUMBER = 30

def getInputData(filename):
    _data = np.genfromtxt(filename, delimiter = ',')
    _data_date = 0;
    _X = [] #Stock information for each period. [P1[s0001,s0002,s0003,...s0015,s0101,s0102,...s0115,...s3001, ...,s3015], P2[], P3[]...]
    _Y = [] #Price trend for next week for each peroid. [P1[-1, -1, 1,1,1,0,-1,...i30],P2[],P3[]...]
    _x = []
    _y = []
    _Z = [] #store the value of percent_change_next_weeks_price
    _z = []

    
    #print('_data=', _data)
    for i in range(1, len(_data)): # Range from 1 to ignore first heading row.
        if(_data_date != _data[i][2]):  #if the data is different date, create a new row for them.
            _data_date = _data[i][2]  #Get date information
            if(len(_x) != 0) :
                _X.append(_x)
                _Y.append(_y)
                _Z.append(_z)
            else:
                pass
            
            _x = list(_data[i][:RECORD_NUMBER])
            if(_data[i][-1] > 0): # if stock price increase in next week.
                _y = [1]
                _z = [_data[i][-1]]
            elif (_data[i][-1] < 0): # if stock price decrease in next week.
                _y = [-1]
                _z = [_data[i][-1]]
            else:  # if stock price keep flat in next week.
                _y = [0]
                _z = [_data[i][-1]]
        else:
            _x.extend(_data[i][:RECORD_NUMBER])
            if(_data[i][-1] > 0): #last data field is label, percent_change_next_weeks_price
                _y.extend([1])
                _z.extend([_data[i][-1]])
            elif (_data[i][-1] < 0):
                _y.extend([-1])
                _z.extend([_data[i][-1]])
            else:
                _y.extend([0])  
                _z.extend([_data[i][-1]])    
    _X.append(_x)
    _Y.append(_y)      
    _Z.append(_z)          
    return np.array(_X), np.array(_Y), np.array(_Z)

def getInputDataMaxY(filename):
    _data = np.genfromtxt(filename, delimiter = ',')
    _data_date = 0;
    _X = [] #Stock information for each period. [P1[s0001,s0002,s0003,...s0015,s0101,s0102,...s0115,...s3001, ...,s3015], P2[], P3[]...]
    _Y = [] #Price trend for next week for each peroid. [P1[-1, -1, 1,1,1,0,-1,...i30],P2[],P3[]...]
    _x = []
    _y = []
    _Z = [] #store the value of percent_change_next_weeks_price
    _z = []    
    _idx = 0
    _max_idx = 0
    _max_y = 0
    
    #print('_data=', _data)
    for i in range(1, len(_data)): # Range from 1 to ignore first heading row.
        if(_data_date != _data[i][2]):  #if the data is different date, create a new row for them.
            _data_date = _data[i][2]  #Get date information
            if(len(_x) != 0) :
                _y[_max_idx] = 1
                _max_idx = 0
                _max_y = 0
                _idx = 0                
                _X.append(_x)
                _Y.append(_y)
                _Z.append(_z)                
            else:
                pass
            
            _x = list(_data[i][:RECORD_NUMBER])
            if(_data[i][-1] > 0 and _data[i][-1] > _max_y): # if stock price increase in next week and the percentage is higher than other stocks.
                _max_y = _data[i][-1]
                _max_idx = _idx
            else: pass
            _y = [0]
            _z = [_data[i][-1]]            
            _idx += 1
        else:
            _x.extend(_data[i][:RECORD_NUMBER])
            if(_data[i][-1] > 0 and _data[i][-1] > _max_y): # if stock price increase in next week and the percentage is higher than other stocks.
                _max_y = _data[i][-1]
                _max_idx = _idx
            else: pass
            _y.extend([0])
            _z.extend([_data[i][-1]])    
            _idx += 1
    _y[_max_idx] = 1 #label for last week of data
    _X.append(_x)
    _Y.append(_y)    
    _Z.append(_z)    
    return np.array(_X), np.array(_Y), np.array(_Z)

def getInputData4DXMax2DY(filename):
    #[data_index, y=each stock, x=stock features, channel=1=each value]
    _data = np.genfromtxt(filename, delimiter = ',')
    _data_date = 0;
    _X = [] #Stock information for each period. [P1[s0001,s0002,s0003,...s0015,s0101,s0102,...s0115,...s3001, ...,s3015], P2[], P3[]...]
    _Y = [] #Price trend for next week for each peroid. [P1[-1, -1, 1,1,1,0,-1,...i30],P2[],P3[]...]
    _x = []
    _y = []
    _Z = [] #store the value of percent_change_next_weeks_price
    _z = []     
    _idx = 0
    _max_idx = 0
    _max_y = 0
    
    #print('_data=', _data)
    for i in range(1, len(_data)): # Range from 1 to ignore first heading row.
        if(_data_date != _data[i][2]):  #if the data is different date, create a new row for them.
            _data_date = _data[i][2]  #Get date information
            if(len(_x) != 0) :
                _y[_max_idx] = 1
                _max_idx = 0
                _max_y = 0
                _idx = 0                
                _X.append(_x)
                _Y.append(_y)
                _Z.append(_z)                 
                _x = []
            else:
                pass
            _x.append(_data[i][:RECORD_NUMBER].reshape(RECORD_NUMBER, 1))
            if(_data[i][-1] > 0 and _data[i][-1] > _max_y): # if stock price increase in next week and the percentage is higher than other stocks.
                _max_y = _data[i][-1]
                _max_idx = _idx
            else: pass
            _y = [0]
            _z = [_data[i][-1]]
            _idx += 1
        else: #all stock in the same date/week
            _x.append(_data[i][:RECORD_NUMBER].reshape(RECORD_NUMBER, 1))
            if(_data[i][-1] > 0 and _data[i][-1] > _max_y): # if stock price increase in next week and the percentage is higher than other stocks.
                _max_y = _data[i][-1]
                _max_idx = _idx
            else: pass
            _y.extend([0])
            _z.extend([_data[i][-1]])
            _idx += 1
    _y[_max_idx] = 1 #label for last week of data
    _X.append(_x)
    _Y.append(_y)
    _Z.append(_z)    
    return np.array(_X), np.array(_Y), np.array(_Z)

def GetPercentChangeNW(index_iter, percent_change_list):
    percentage_list = []
    avg_percentage = 0.0
    sum_percentage = 0.0
    
    for i, idx in enumerate(index_iter):
        percentage_list.append(percent_change_list[i][idx])
        sum_percentage += percent_change_list[i][idx]
    avg_percentage = sum_percentage/i
    return percentage_list, avg_percentage, sum_percentage

MDX, MDY, MDZ = getInputData4DXMax2DY(FILE_NAME)
X, Y, Z = getInputDataMaxY(FILE_NAME)
print('length of X:%d, X[0]:%d'%(len(X), len(X[0])))
print('shape of MDX, MDY, MDZ', MDX.shape, MDY.shape, MDZ.shape)
#MDX, MDY = getInputData4DXMax2DY(FILE_NAME)
#X, Y = getInputDataMaxY(FILE_NAME)
#print('shape of MDX, MDY', MDX.shape, MDY.shape)
#print('X=', X)
#print('Y=', Y)

TOTAL_SIZE = len(X)
TESTDATA_SIZE = 13
VALIDATION_SIZE = 0

train_data = X[:TOTAL_SIZE-TESTDATA_SIZE-VALIDATION_SIZE]
train_labels = Y[:TOTAL_SIZE-TESTDATA_SIZE-VALIDATION_SIZE]
train_next_week_percent = Z[:TOTAL_SIZE-TESTDATA_SIZE-VALIDATION_SIZE]

validation_data = X[TOTAL_SIZE-TESTDATA_SIZE-VALIDATION_SIZE:TOTAL_SIZE-TESTDATA_SIZE]
validation_labels = Y[TOTAL_SIZE-TESTDATA_SIZE-VALIDATION_SIZE:TOTAL_SIZE-TESTDATA_SIZE]
validation_next_week_percent = Z[TOTAL_SIZE-TESTDATA_SIZE-VALIDATION_SIZE:TOTAL_SIZE-TESTDATA_SIZE]

test_data = X[TOTAL_SIZE-TESTDATA_SIZE:]
test_labels = Y[TOTAL_SIZE-TESTDATA_SIZE:]
test_next_week_percent = Z[TOTAL_SIZE-TESTDATA_SIZE:]

train_size = train_labels.shape[0]
test_size = test_labels.shape[0]
print('TOTAL_SIZE', TOTAL_SIZE)
print('Train size', train_size, 'training label shape', train_labels.shape)
print('Validation size', validation_data.shape)
print('Test size', test_size)
#    print('train_data', train_data)
#    print('train_labels', train_labels)
#    print('validation_data', validation_data)
#    print('validation_labels', validation_labels)

# Parameters
learning_rate = 0.001
training_iters = 4000
batch_size = 1
display_step = 1000

# Network Parameters
n_random_seed = 4
n_mean = 0
n_stddev = 1
n_input = RECORD_NUMBER # data input
n_steps = STOCK_NUMBER # timesteps
n_hidden = 256 # hidden layer num of features
n_classes = STOCK_NUMBER # classes (0-29 stocks)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])


# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=n_mean, stddev=n_stddev, seed=n_random_seed))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes], mean=n_mean, stddev=n_stddev, seed=n_random_seed))
}

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # Old version syntax
    #x = tf.split(0, n_steps, x)
    x = tf.split(x, n_steps, 0)

    # Define a lstm cell with tensorflow
    # Old version syntax
    #lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_cell = rnn.BasicLSTMCell(n_hidden)

    # Get lstm cell output
    # Old version syntax
    #outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name='GradientDescent').minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
if (tf.__version__ <= '0.11.0'):
    # Initializing the variables for r.011
    init = tf.initialize_all_variables()
else:
    # Initializing the variables for r0.12
    init = tf.global_variables_initializer()

# Launch the graph
test_data = np.reshape(test_data, (-1, n_steps, n_input))
test_labels = np.reshape(test_labels, (-1, n_classes))
train_data = np.reshape(train_data, (-1, n_steps, n_input))
train_labels = np.reshape(train_labels, (-1, n_classes))
    
with tf.Session() as sess:
    sess.run(init)
    step = 0
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x = np.reshape(train_data[step*batch_size:(step+1)*batch_size], (-1, n_steps, n_input))
        batch_y = np.reshape(train_labels[step*batch_size:(step+1)*batch_size], (-1, n_classes))
        
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 30 seq of 15 elements
        #batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print ("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
#     test_len = 128
#     test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
#     test_label = mnist.test.labels[:test_len]
#     print "Testing Accuracy:", \
#         sess.run(accuracy, feed_dict={x: test_data, y: test_label})

    result_raw = tf.cast(pred, "float")
    result_stock_index = tf.argmax(result_raw, 1) 
    Y_label_index = tf.argmax(y, 1)
    
    print ('==>Training Data Accuracy:', accuracy.eval({x: train_data, y: train_labels}))
    #print ('Training Data Raw Result:', result_raw.eval({x: train_data}))
    result_stock_index_list = result_stock_index.eval({x: train_data})
    Y_label_index_list = Y_label_index.eval({y:train_labels})
    print ('Training Data Max. Stock Index Predict Result:', result_stock_index_list)
    print ('Training Y labels:', Y_label_index_list )
    max_percentage_list, avg_max_p, sum_max_p = GetPercentChangeNW(result_stock_index_list, train_next_week_percent)
    print ('Percentage Change next week by Predict', max_percentage_list)
    print ('=>Average percentage Change next week by Predict is %f, (Sum of percentage: %f)'%( avg_max_p, sum_max_p))
    max_percentage_list, avg_max_p, sum_max_p = GetPercentChangeNW(Y_label_index_list, train_next_week_percent)
    print ('Percentage Change next week by label', max_percentage_list)
    print ('Average percentage Change next week by label is %f, (Sum of percentage: %f)'%( avg_max_p, sum_max_p))
    
    # Testing section
    print ('==>Testing Data Accuracy:', accuracy.eval({x: test_data, y: test_labels}))
    result_stock_index_list = result_stock_index.eval({x: test_data})
    Y_label_index_list = Y_label_index.eval({y:test_labels})
    print ('Testing Data Max. Stock Index Predict Result:', result_stock_index_list)
    print ('Testing Y labels:', Y_label_index_list )
    
    max_percentage_list, avg_max_p, sum_max_p = GetPercentChangeNW(result_stock_index_list, test_next_week_percent)
    print ('Percentage Change next week by Predict', max_percentage_list)
    print ('=>Average percentage Change next week by Predict is %f, (Sum of percentage: %f)'%( avg_max_p, sum_max_p))
    max_percentage_list, avg_max_p, sum_max_p = GetPercentChangeNW(Y_label_index_list, test_next_week_percent)
    print ('Percentage Change next week by label', max_percentage_list)
    print ('Average percentage Change next week by label is %f, (Sum of percentage: %f)'%( avg_max_p, sum_max_p))