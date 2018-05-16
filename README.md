# Adaptive Deep LSTM Ensemble Method (ADLE)

## Immediate TODO

+ Add `Aggregator` as an ensemble parameter. Define a base class with methods that an aggregator must have, aggregator instances, e.g. kNN, ANN, etc., will inherit this class to aggregate the ensemble's predictions

+ Add changepoint detection to get ensemble bounds (python rupture library (DynP and PELT))

## Overview

An ensemble method in which a set of LSTM-RNNs are trained on some subset of the data to approximate different functions. The problem with complex,
non-stationary data is that the function generating the data changes throughout time. That is, the statistical properties of the data generating
function are non-constant. This approach essentially aims to train NNs to model a different function as the data-generating function changes through time. An aggregating method,
such as another NN or k-nearest neighbors, is used to get the predictions of future values.

In this case, each neural network is a dictionary entry which ties its data segment's statistical properties to itself (key=statistical properties, value=NN).

## The Paper

The paper can be found in ./paper/ with name predicting-future-events.pdf, that is the full name and filepath is ./paper/predicting-future-events.pdf

## Usage

### Easy Usage

If wanting to run the ensemble standalone see **Standalone Ensemble** below. This example run allows a quick run of the example datasets.

Running **python adle.py** with no parameters will prompt the user for the required parameters. Otherwise
the user can supply any number of parameters and the rest will be filled in manually. The required parameters are,

+ 'data' : Determines which dataset to use from the options (1=Sunspots, 2=EUR/USD Exchange Rate, 3=Mackey-Glass)
+ 'ensemble' : Takes one of the values ('t', 'l'). Setting this value to 't' will cause each
ensemble method to be trained at runtime. Supplying 'l' results in pretrained weights being loaded from a file.
+ 'run_bencmarks' : Takes one of the values ('y', 'n'). Setting to 'y' runs the benchmark LSTM to compare
error rates (mean squared error).
+ 'benchmark' : Takes the values ('t', 'l'). Same as the 'ensemble' parameter but loads or trains the
weights of the benchmark LSTM at runtime.

Additionally, if invalid parameter values are supplied via command line the user will be prompted to enter these manually. Parameters supplied that aren't listed above are ignored.

Examples:
  + python3 adle.py data=1 ensemble=t run_bencmarks=y benchmark=l
    + Runs ADLE on the Sunspots dataset while traning the ensemble weights and loading the benchmark weights from ./weights/benchmarks/sunspots.h5
  + python3 adle.py data=3
    + Runs ADLE on the Mackey-Glass equation output. The user will be prompted to enter the other parameters.
  + python3 adle.py data=2 run_bencmarks=n
    + Runs ADLE on the EUR/USD forex dataset but does not create and compare the benchmark LSTM. The user will be prompted to input the 'ensemble' parameter.

Does not run the ARIMA benchmark.

### Standalone Ensemble

The file **ensemble_example.py** is an example of how to use the ensemble method standalone.

This can be done with any univariate time series dataset where the first column is the timestamp or timestep and the second column is the time series values.

The code is fairly well documented which shows other potential parameters for the ensemble, but the most common parameters are demonstrated in the example file.

### Datasets

Because these datasets are univariate they tend to be small in terms of file size. Thus the datasets used for the NIPS report are included in the GitHub repositoty under ./data/

### Requirements

+ Python 3.x (specifically tested with 3.6.1 and 3.5.2)
+ Python Libraries
  + Tensorflow
  + Pandas
  + statsmodels
  + matplotlib
  + scikit-learn (sklearn)
  + Keras
  + h5py (loading and saving weights)
  + numpy

### Project Structure

+ ./graphs, ./data_tools, ./old_proto - can essentially be ignored as they were used to create the paper, look at the datasets, and prototype the ensemble.
+ ./paper - contains everything needed for compiling the paper and as well as a PDF copy
+ ./results - contains CSVs of the various parameters used for the ensemble on the three different datasets
+ ./weights - pretrained weights for the ensemble and base LSTM models
+ ./ - example usage of the ensemble and benchmark models


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

+ https://github.com/deepcharles/ruptures/blob/master/ruptures/detection/dynp.py
  + Rupture is changepoint detection for Python (DynP = Segment Neighbors)

+ https://arxiv.org/pdf/1101.1438.pdf
  + PELT algorithm 
  
+ https://epublications.marquette.edu/cgi/viewcontent.cgi?referer=https://scholar.google.com/&httpsredir=1&article=1436&context=theses_open
  + Uses an ensemble of different learners with weighted predictions from the ensemble
