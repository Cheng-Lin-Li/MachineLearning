# Machine Learning
Implementations of machine learning algorithm by Python 3

The repository provides demo programs for implementations of basic machine learning algorithms by Python 3. I hope these programs will help people understand the beauty of machine learning theories and implementations.

I will enrich those implementations and descriptions from time to time. If you include any of my work into your website or project; please add a link to this repository and send me an email to let me know.

Your comments are welcome.
Thanks,

|Algorithm|Description|Link|
|------|------|--------|
|[Decision Tree](https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/DecisionTree)|By measuring information gain via calculating the entropy of previous observations, decision tree algorithm may help us to predict the decision or results|[Specification](https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/DecisionTree/TechnicalSpecification-%5BDecisionTree%5D-%5B1.1%5D-%5B20160929%5D.pdf) and [Source Code](https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/DecisionTree/DecisionTree.py)|
|[Fast Map](https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/FastMap)|An approach for dimensions reductions|[Specification](https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/FastMap/TechnicalSpecification-%5BPCA_FastMap%5D-%5B1.0%5D-%5B20160929%5D.pdf) and [Source Code](https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/FastMap/FastMap.py)|
|[Gaussian Mixture Models (GMMs)](https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/GMM)|GMMs are among the most statistically mature methods for data clustering (and density estimation)|[Specification](https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/GMM/INF552-TechnicalSpecification-%5Bk-means_EM-GMM%5D-%5B1.2%5D-%5B20170515%5D.pdf) and [Source Code](https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/GMM/GMM.py)|
|[Hierarchical clustering (HAC)](https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/HAC)|HCA seeks to build a hierarchy of clusters from bottom up or top down. This is a bottom up implementation.|[Source Code](https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/HAC/HAC.py)|
|[Hidden Markov model (HMM) and Viterbi](https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/HMM)|HMM is a statistical Markov model in which the system being modeled is assumed to be a Markov process with unobserved (i.e. hidden) states. The Viterbi algorithmis used to compute the most probable path (as well as its probability). It requires knowledge of the parameters of the HMM model and a particular output sequence and it finds the state sequence that is most likely to have generated that output sequence. It works by finding a maximum over all possible state sequences. In sequence analysis, this method can be used for example to predict coding vs non-coding sequences.|[Specification](https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/HMM/INF552-TechnicalSpecification-%5BHMM%5D-%5B1.0%5D-%5B20161203%5D.pdf) and [Viterbi Algorithm Source Code](https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/HMM/HMM-Viterbi.py)|
|[K-Means](https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/K-Means)|One of most famous and easy to understand clustering algorithm|[Source Code](https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/K-Means/K-Means.py)|
|[Neural Network](https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/NeuralNetwork)|The foundation algorithm of deep learning|[Specification](https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/NeuralNetwork/INF552-TechnicalSpecification-%5BNeuralNetwork%5D-%5B1.0%5D-%5B20161104%5D.pdf) and [Source Code](https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/NeuralNetwork/NeuralNetwork.py)|
|[PCA](https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/PCA)|An algorithm for dimension reductions. PCA is a statistical technique, via orthogonal transformation, convert dataset that some of them may correlated to a new data space that is linearly uncorrelated set of values. This new set of data call principal components. PCA is sensitive to the relative scaling of the original variables, so before applying PCA, data pre-processing step is very important and we should always do. Mean normalization (x - mean of the feature) or feature scaling (x - mean)/max(x) or (x-mean)/(Standard deviation of x) are required. |[Specification](https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/PCA/INF552-TechnicalSpecification-PCA_FastMap-%5B1.0%5D-%5B20161011%5D.pdf) and [Source Code](https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/PCA/PCA.py)|
|[Neural Network and Long Short Term Memory (LSTM) on Tensorflow](https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/TensorFlow)|This is a project which implemented Neural Network and Long Short Term Memory (LSTM) to predict stock price in Python 3 by Tensorflow|[Project Report](https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/TensorFlow/ProjectReport.pdf) and [MLP Source Code](https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/TensorFlow/StockPriceForecasting-MLP.py), [LSTM Source Code](https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/TensorFlow/StockPriceForecasting-LSTM.py)|



## Reference:
* Machine Learning at Coursera: https://www.coursera.org/learn/machine-learning/
* Machine Learning Course CS229 at Stanford: https://see.stanford.edu/Course/CS229
* Machine Learning Foundations at NTU: https://www.youtube.com/watch?v=nQvpFSMPhr0&list=PLXVfgk9fNX2I7tB6oIINGBmW50rrmFTqf
* Machine Learning Techniques at NTU: https://www.youtube.com/watch?v=A-GxGCCAIrg&list=PLXVfgk9fNX2IQOYPmqjqWsNUFl2
* Hidden Markov model (HMM): https://en.wikipedia.org/wiki/Hidden_Markov_model
* Viterbi Algorithm: http://homepages.ulb.ac.be/~dgonze/TEACHING/viterbi.pdf
* LSTM: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
* PCA at wikipedia: https://en.wikipedia.org/wiki/Principal_component_analysis


Cheng-Lin Li@University of Southern California

emailto: chenglil@usc.edu or clark.cl.li@gmail.com
