# This is an implementation of MLP and LSTM by TensorFlow 1.1 in Python 3

## Project:
The one of the most challenging issue is stock price or indices prediction in the financial industry. On the other hand, machine learning and big data techniques in vision recognition has matured considerably over the last decade. This research adopts Multi-Layer Perceptron (MLP) and Long-Short Term Memory (LSTMs) neural networks to compete with Dynamic-radius Species-conserving Genetic Algorithm (DSGA) for short term stock price prediction. The result indicates that MLP may have a better potential than DSGA on short term stock price prediction and that LSTMs may require more training data to surpass DSGA. 


### Usage: python StockPriceForecasting.py (or StockPriceForecasting-LSTM.py)	

### Input: A text file 'dow_jones_index.csv'.
The text file include 750 data records.  360 are from the first quarter of the year (Jan to Mar). 390 are from the second quarter of the year (Apr to Jun).

### Output: Iterations and training / testing data results of selected stock index prediction.

### Warning:

1. The program just converted from TensorFlow r0.12 to r1.1. Potential defects may exist.
2. The program output need to be reviewed again. This is only a reference code for you to deal with your own data and apply to your own application.

## Reference 
  1. Google TensorFlow, MNIST For ML Beginners, https://www.tensorflow.org/get_started/mnist/beginners
  2. Google TensorFlow, Deep MNIST for Experts, https://www.tensorflow.org/get_started/mnist/pros
  3. Donahue, J., Hendricks, L.A., Guadarrama, S., Rohrbach, M., Venugopalan, S., Darrell, T. And Saenko, K. 2015. Long-term recurrent convolutional networks for visual recognition and description. In 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 7-12 June 2015, Anonymous IEEE, Piscataway, NJ, USA, 2625-34., https://arxiv.org/abs/1411.4389v3
  4. Brown, M.S., Pelosi, M.J. AND Dirska, H. 2013. Dynamic-radius Species-conserving Genetic Algorithm for the Financial Forecasting of Dow Jones Index Stocks. In 9th International Conference, MLDM 2013, 19-25 July 2013, Anonymous Springer Verlag, Berlin, Germany, 27-41., https://link.springer.com/chapter/10.1007/978-3-642-39712-7_3
  5. colah's blog, Understanding LSTM Networks, http://colah.github.io/posts/2015-08-Understanding-LSTMs/
  6. Deep Learning for Java, A Beginner’s Guide to Recurrent Networks and LSTMs, https://deeplearning4j.org/lstm.html
