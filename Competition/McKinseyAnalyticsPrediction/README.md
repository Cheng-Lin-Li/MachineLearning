# Data Competition of Machine Learning by scikit-learn and Keras library
Implementations of machine learning algorithms by Python 3 for some selected data competition.

The folders included demo programs for leverage scikit-learn, Keras libraries to challenge datathon with Python 3. 

|Algorithm|Description|Link|
|------|------|--------|
|[Support Vector Regression (SVR), Neural Network, Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU)](https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/Competition/McKinseyAnalyticsPrediction) | This task is to perform prediction for number of vehicles by given data. This is a demo program to leverage mutile models from existing libraries in one challenge. The final result can be improved by some emsemble techniques like Bootstrap aggregating (bagging), boosting, and stacking to get better performance.|[Source Code](https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/Competition/McKinseyAnalyticsPrediction/NumberOfVehiclesPrediction.ipynb)|


## The task:
### Prediction Number of Vehicles
Given 20 months Date time, ID of junctions, and number of Vehicles to predict number of vehicles in next 4 months.

Training Data:

|Data field|Description|
|---|---|
|DateTime|Date time info|
|Junction|Junction no.|
|Vehicles|number of vehicles|
|ID| Data unique ID|

Test Data:

|Data field|Description|
|---|---|
|DateTime|Date time info|
|Junction|Junction no.|
|ID| Data unique ID|