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
    adapt = True
    if dataset == 's':
        ensemble = Ensemble(train_style='overlap', num_segments=5,
                            load_train=control.ensemble, )
        adapt = False
    elif dataset == 'e':
        ensemble = ...
    else:
        ensemble = ...
    ensemble.train_methods()
    testy = ensemble.testy
    predictions = ensemble.get_predictions(adaptive=adapt)

    # Run the benchmark LSTM if need be

    # Compare the errors of the ensemble and the LSTM
    if control.run_benchmarks:
        


if __name__ == '__main__':
    """
        Main control for running ADLE.
    """
    global control
    control = Control()
    parse_params(sys.argv)
    fill_params()
    run()
