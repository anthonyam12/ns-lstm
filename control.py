"""
	Holds parameters for the control of the example run of ADLE (adle.py).

	Author: Anthony Morast
	Date: 4/23/2018
"""

class Control(object):
    def __init__(self):
        """
            Controls for running ADLE.
        """
        self.dataset = None
        self.ensemble = None
        self.benchmarks = None
        self.run_benchmarks = None

        self.MACKEY = 3
        self.EURUSD = 2
        self.SUNSPOTS = 1
        self.data_options = [self.MACKEY, self.EURUSD, self.SUNSPOTS]
        self.valid_load_train = ['t', 'T', 'l', 'L']
        self.valid_run_benchmarks = ['y', 'n', 'Y', 'N']

    def set_dataset(self, option):
        """
            Dataset selection setter.
        """
        self.dataset = int(option)

    def get_dataset(self):
        """
            Dataset selection getter.
        """
        return self.dataset


    def set_ensemble_load(self, load_or_train):
        """
            Ensemble load or train setter.
        """
        self.ensemble = load_or_train

    def get_ensemble_load(self):
        """
            Ensemble load or train getter.
        """
        return self.ensemble


    def set_benchmark_load(self, load_or_train):
        """
            Benchmarks load or train setter.
        """
        self.benchmarks = load_or_train

    def get_benchmark_load(self):
        """
            Benchmarks load or train getter.
        """
        return self.benchmarks

    def set_run_benchmark(self, run_benchmarks):
        """
            Benchmarks load or train setter.
        """
        bm = False
        if run_benchmarks.lower() == 'y':
            bm = True
        self.run_benchmarks = bm

    def get_run_benchmark(self):
        """
            Benchmarks load or train getter.
        """
        return self.run_benchmarks
