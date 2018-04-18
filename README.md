# Adaptive Deep LSTM Ensemble Method (ADLE)
An ensemble method in which a set of LSTM-RNNs are trained on some subset of the data to approximate different functions. The problem with complex, 
non-stationary data is that the function generating the data changes throughout time. That is, the statistical properties of the data generating
function are non-constant. This approach essentially aims to train NNs to model a different function as the the data-generating function changes through time. An aggregating method,
such as another NN or k-nearest neighbors, is used to get the predictions of future values. 

In this case, each neural network is an entry dictionary which ties its data segment's statistical properties to itself (key=statistical properties, value=NN).

--------

# Description

The main idea of ADLE, is that changes in statistical properties over different time periods will be captured by training networks on different data segments. This will create an ensemble of *n* networks all trained with different time periods and (potentially) approximating different, but sometimes similar, functions.

--------

# Research Questions

+ Can the method outperform other popular methods when viewing the problem as a regression problem?
+ Does the method perform better when modeling magnitudes of changes or directions of change?
  + -1 downward trend, 0 no change, 1 upward trend
  + "The value tomorrow will be approx. .0005 units higher than it was today"
+ Does the model perform well on mutlivariate time series?
+ Can change detection techniques be used to find the proper bounds of each LSTM's segment of data?
+ Can this have a parallel implementation?
+ Do other models work better in the ensemble?
  + ConvLSTM
  + CNN
  + ANN
  + SVM
+ Can the hyperparameters for each network be estimated based on the complexity of the data segment being used to train that network?
  + A smaller, more narrow may be best for sections of the data that are more linear while deeper, wider netowrks might be effective for more complex segments
  + Can also lead to faster training times
+ Will preprocessing the datasets affect the performance of the method?

--------

# Relevant Materials 

Below is a list of papers and other material relevant to this project/research.

+ Zhang, Xiru, and Jim Hutchinson. "Simple architectures on fast machines: practical issues in nonlinear time series prediction." (1994): 219-241. (Time series prediction, Sante Fe Institute book)
