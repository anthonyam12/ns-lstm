# Adaptive Deep LSTM Ensemble Method (ADLE)

## Overview

An ensemble method in which a set of LSTM-RNNs are trained on some subset of the data to approximate different functions. The problem with complex,
non-stationary data is that the function generating the data changes throughout time. That is, the statistical properties of the data generating
function are non-constant. This approach essentially aims to train NNs to model a different function as the data-generating function changes through time. An aggregating method,
such as another NN or k-nearest neighbors, is used to get the predictions of future values.

In this case, each neural network is a dictionary entry which ties its data segment's statistical properties to itself (key=statistical properties, value=NN).


## Usage

Running 'python adle.py' with no parameters will prompt the user for the required parameters. Otherwise
the user can supply any number of parameters and the rest will be filled in manually. The required parameters
are,

+ 'data' : Determines which dataset to use from the options (1=Sunspots, 2=EUR/USD Exchange Rate, 3=Mackey-Glass)
+ 'ensemble' : Takes one of the values ('t', 'l'). Setting this value to 't' will cause each
ensemble method to be trained at runtime. Supplying 'l' results in the weights being loaded from a file.
+ 'run_bencmarks' : Takes one of the values ('y', 'n'). Setting to 'y' runs the benchmark LSTM to compare
error rates (mean squared error).
+ 'benchmark' : Takes the values ('t', 'l'). Same as the 'ensemble' parameter but loads or trains the
weights of the benchmark LSTM at runtime.

Additionally, if invalid parameters are supplied via command line the user will be prompted to enter these
manually. Parameters supplied that aren't listed above are ignored.

Does not run the ARIMA benchmark.

## Requirements

+ Python 3.x (specifically tested with 3.6.1)
+ Python Libraries
  + Tensorflow
  + Pandas
  + statsmodels
  + matplotlib
  + scikit-learn (sklearn)
  + Keras
  + h5py (loading and saving weights)
  + numpy


## Research Questions

+ Can the method outperform other popular methods when viewing the problem as a regression problem?
+ Does the method perform better when modeling magnitudes of changes or directions of change?
  + -1 downward trend, 0 no change, 1 upward trend
  + "The value tomorrow will be approx. .0005 units higher than it was today"
+ Does the model perform well on multivariate time series?
+ Can change detection techniques be used to find the proper bounds of each LSTM's segment of data?
+ Can this have a parallel implementation?
+ Do other models work better in the ensemble?
  + ConvLSTM
  + CNN
  + ANN
  + SVM
+ Can the hyperparameters for each network be estimated based on the complexity of the data segment being used to train that network?
  + A smaller, more narrow network may be best for sections of the data that tend towards linearity while deeper, wider networks might be effective for more complex segments
  + Can also lead to faster training times
+ Will preprocessing the datasets affect the performance of the method?

--------

## Relevant Materials

Below is a list of papers and other material relevant to this project/research.

+ Zhang, Xiru, and Jim Hutchinson. "Simple architectures on fast machines: practical issues in nonlinear time series prediction." (1994): 219-241. (Time series prediction, Sante Fe Institute book)
