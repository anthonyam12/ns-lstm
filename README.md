# Non-Stationary LSTM Ensemble (NS-LSTM)
An ensemble method where each ANN is trained on a subset of the data. The prediction is based on each networks (weighted) predictions.

OR

An ensemble method in which a set of ANNs are trained on some subset of the data to approximate different functions. The problem with complex, 
non-stationary data is that the function generating the data changes throughout time. That is, the statistical properties of the data generating
function are non-constant. This approach essentially aims to train NNs to model a different function as the the data-generating function changes through time. An aggregating method,
such as another NN or k-nearest neighbors, is used to get the predictions of future values. 

In this case, each neural network will be in a dictionary which ties its statistical properties to itself (key=statistical properties, value=NNs).

--------

# Description

The main idea of NS-LSTM, is the local statistical properties over certain time periods will be captured by training networks in different locales. This will create an ensemble of *n* networks all trained with over different time periods and (potentially) approximating different, but sometimes similar, functions.

The ensemble alone should provide better predictions than regular neural networks and other econometric models. One extension would be to weight the predictions of the neural networks based on the variance of the time period they model. For instance, if each network is trained over some 30-day period, the variance of the values over that period will be computed. This variance will be tied to the neural network that was trained. In the prediction process a **GARCH** model is used to track the current measure of local variance. ANN's with variance measures close to the current local variance are weighted higher than the networks with more different variances. 

--------

# Research Goals

+ Can the method outperform other popular methods when viewing the problem as a regression problem?
+ Does the method perform better when modeling magnitudes of changes or directions of change?
  + -1 downward trend, 0 no change, 1 upward trend
  + "The value tomorrow will be approx. .0005 units higher than it was today"

--------

# Relevant Materials 

Below is a list of papers and other material relevant to this project/research.

+ Robust maximum likelihood training of heteroscedastic probabilistic
neural networks; Zheng Rong Yanga, Sheng Chenb; Journal: Neural Networks; Year: 1998; Found at: https://eprints.soton.ac.uk/251026/1/nn-els98.pdf

+ Zhang, Xiru, and Jim Hutchinson. "Simple architectures on fast machines: practical issues in nonlinear time series prediction." (1994): 219-241. (Time series prediction, Sante Fe Institute book)

Search Terms

+ heteroscedastic neural network; scholar.google.com
