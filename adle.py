"""
	An example run of ADLE in which the user can select to use 1 of the 3
    test datasets, train or load the weights, and compare mse/mae with a 
	benchmark LSTM network. 

	Parameters are not required via command line but can be supplied based on 
	the specifications in README.md. If parameters are not supplied the user 
	will be prompt to input them. 

	Author: Anthony Morast 
	Date: 4/23/2018
"""

from data_handler import *
from ensemble import *
from lstm import *
from errors import *
from control import *

import sys
import os

def valid_load(value, type):
    """
        Returns true if valid value for loading or training is supplied.
    """
    control = Control() # only need for statics
    if not value in control.valid_load_train:
        print("Warning: Invalid option for load/train for", type, "valid options are {'t', 'l'}")
        return False
    return True


def valid_dataset(value):
    """
        Determine if the dataset parameter supplied is valid.
    """
    try:
        value = int(value)
    except ValueError:
        print("Warning: Invalid dataset selection, valid options are {1=sunspots, 2=eur/usd, 3=mackey}.")
        return False
    if not value in control.data_options:
        print("Warning: Invalid dataset selection, valid options are {1=sunspots, 2=eur/usd, 3=mackey}.")
        return False
    return True


def valid_run_benchmarks(value):
    """
        Determines if benchmark run parameter is valid.
    """
    if not value in control.valid_run_benchmarks:
        print("Warning: Invalid parameter for run_benchmarks, valid options are {'y', 'n'}.")
        return False
    return True


def parse_params(argv):
    """
        Parses the command line parameters setting them in the control class.
    """
    global control
    for arg in argv:
        if 'data' in arg:
            option = arg[-1]
            if valid_dataset(option):
                control.set_dataset(option)
        elif 'ensemble' in arg:
            option = arg[-1]
            if valid_load(option, 'ensemble'):
                control.set_ensemble_load(option)
        elif 'benchmark' in arg:
            option = arg[-1]
            if valid_load(option, 'benchmarks'):
                control.set_benchmark_load(option)
        elif 'run_benchmarks' in arg:
            option = arg[-1]
            if valid_run_benchmarks(option):
                control.set_run_benchmark(option)
        elif 'adle' in arg:
            continue
        else:
            print("Warning: argument", arg, "is not a valid argument and will be ignored.")


def fill_params():
    """
        Prompts user for parameters not supplied via command line.
    """
    global control
    if control.get_run_benchmark() is None:
        option = input("Run benchmarks (y/n)? ")
        while not valid_run_benchmarks(option):
            option = input("Invalid. Run benchmarks (y/n)? ")
        control.set_run_benchmark(option)
    if control.get_benchmark_load() is None and control.get_run_benchmark():
        option = input("Load or train benchmark model weights (t/L): ")
        while not valid_load(option, 'benchmark'):
            option = input("Invalid. Load or train benchmark model weights (t/L): ")
        control.set_benchmark_load(option)
    if control.get_dataset() is None:
        option = input("Select a dataset (1 = Sunspots, 2 = EUR/USD, 3 = Mackey-Glass): ")
        while not valid_dataset(option):
            option = input("Invalid. Select a dataset (1 = Sunspots, 2 = EUR/USD, 3 = Mackey-Glass): ")
        control.set_dataset(option)
    if control.get_ensemble_load() is None:
        option = input("Load or train ensemble model weights (t/L): ")
        while not valid_load(option, 'ensemble'):
            option = input("Invalid. Load or train ensemble model weights (t/L): ")
        control.set_ensemble_load(option)


def run():
    """
        Runs the program per user specification.
    """
    global control
    data_file = ''
    dataset = ''
    if control.dataset == 1:
        data_file = './data/Sunspots.csv'
        dataset = 's'
    elif control.dataset == 2:
        data_file = './data/EURUSD.csv'
        dataset = 'e'
    else:
        data_file = './data/mackey.csv'
        dataset = 'm'

    # Run the ensemle
    ensemble = None
    # different datasets have different hyperparameters
    print("Running ensemble method...")
    adapt = True
    if dataset == 's':
        ensemble = Ensemble(train_style='overlap', num_segments=5,
                            load_train=control.ensemble, base_size=1000,
                            trainsize=2000, dataset='s')
        ensemble.set_data_from_file('./data/Sunspots.csv')
        ensemble.create_datasets()
        ensemble.create_methods(batch_size=100, epochs=1850, verbose=0, params='s')
        adapt = False
    elif dataset == 'e':
        ensemble = Ensemble(train_style='overlap', num_segments=5,
                            load_train=control.ensemble, base_size=1000,
                            trainsize=2000, dataset='e')
        ensemble.set_data_from_file('./data/EURUSD.csv')
        ensemble.create_datasets()
        ensemble.create_methods(batch_size=200, epochs=700, verbose=0, params='e')
    else:
        ensemble = Ensemble(train_style='overlap', num_segments=5,
                            load_train=control.ensemble, base_size=400,
                            trainsize=500, dataset='m')
        ensemble.set_data_from_file('./data/mackey.csv')
        ensemble.create_datasets()
        ensemble.create_methods(batch_size=100, epochs=900, verbose=0, params='m')

    ensemble.train_methods()
    e_testy = ensemble.testy
    e_predictions = ensemble.get_predictions(adaptive=adapt)

    # Run the benchmark LSTM if need be
    dh = None
    print(dataset)
    if control.run_benchmarks:
        print("Running benchmark network...")
        # Get the proper dataets loaded into the data handler
        if dataset == 's':
            dh = DataHandler('./data/Sunspots.csv')
            dh.timeSeriesToSupervised()
            dh.splitData(len(dh.tsdata) - 2000, 2000, 0)
        elif dataset == 'e':
            dh = DataHandler('./data/EURUSD.csv')
            dh.timeSeriesToSupervised()
            dh.splitData(len(dh.tsdata) - 2000, 2000, 0)
        else:
            dh = DataHandler('./data/mackey.csv')
            dh.timeSeriesToSupervised()
            dh.splitData(len(dh.tsdata) - 500, 500, 0)

        # split and reshape the data
        train, test, _ = dh.getDataSets()
        trainx, trainy = train[:, 1], train[:, 3]
        testx, testy = test[:, 1], test[:, 3]
        trainy = trainy.reshape(trainy.shape[0], 1)
        trainx = trainx.reshape(trainx.shape[0], 1)
        testx = testx.reshape(testx.shape[0], 1)
        trainx = trainx.reshape((trainx.shape[0], 1, trainx.shape[1]))
        testx = testx.reshape((testx.shape[0], 1, testx.shape[1]))

        # create the LSTMs and train or load the weights
        lstm = None
        weights_file = ''
        if dataset == 's':
            weights_file = './weights/benchmarks/sunspots.h5'
            lstm = MyLSTM(trainx.shape[1], 11, [32, 35, 41, 13, 32, 50, 34, 7, 38, 23, 50], 1,
                          epochs=750, batch_size=100,
                          fit_verbose=0, variables=trainx.shape[2])
        elif dataset == 'e':
            weights_file = './weights/benchmarks/eurusd.h5'
            lstm = MyLSTM(trainx.shape[1], 7, [8,38,46,31,49,14,14], 1,
                          epochs=300, batch_size=200,
                          fit_verbose=2, variables=trainx.shape[2])
        else:
            weights_file = './weights/benchmarks/mackey.h5'
            lstm = MyLSTM(trainx.shape[1], 9, [28,4,36,47,2,4,5,1,37], 1,
                          epochs=702, batch_size=100,
                          fit_verbose=0, variables=trainx.shape[2])

        if control.get_benchmark_load() == 'l':
            lstm.load_model_weights(weights_file)
        else:
            lstm.train(trainx, trainy)

        b_predictions = lstm.predict(testx)

    # Compare the errors of the ensemble and the LSTM
    print('\n')
    print('Ensemble MSE: ', mse(e_testy, e_predictions))
    print('Ensemble MAE: ', mae(e_testy, e_predictions))
    print('Ensemble SMAPE:', smape(e_testy, e_predictions))
    print('\n')
    if control.get_run_benchmark():
        print('Benchmark MSE: ', mse(testy, b_predictions))
        print('Benchmark MAE: ', mae(testy, b_predictions))
        print('Benchmark SMAPE:', smape(testy, b_predictions))
        print('\n')


if __name__ == '__main__':
    """
        Main control for running ADLE.
    """
    global control
    control = Control()
    parse_params(sys.argv)
    fill_params()
    run()
