from method import *
from data_handler import *
from errors import *
from lstm import *

import numpy as np

class Ensemble(object):
    def __init__(self, train_style='overlap', num_segments=5, base_size=500,
                 trainsize=2000, lookback=1, k=-1, verbose=0):
        """
            train_style - defines how the data will be segmented for the ensemble methods
                         (sequential, overlap, random_segments)
            num_segments - the number of segments for the training data
            base_size - only valid for 'overlap' train style. Determine the size
                        of the training data for each method
            trainsize - number of samples to use for training
            lookback - how many past data points to use as inputs. Usually 1 works best
            k - kNN; number of nearest neighbors to use for prediction
            verbose - determine the level of output (0, 1, 2)
        """
        self.methods = []
        self.train_style = train_style
        self.num_segments = num_segments
        self.base_size = base_size
        self.trainsize = trainsize
        self.look_back = lookback
         # only used for overlap training
        self.shift = int((self.trainsize - self.base_size)/(self.num_segments-1))
        # used for sequential training
        self.segment_size = int(self.trainsize/self.num_segments)
        self.k = k
        if k == -1:
            self.k = num_segments
        self.data_set = False
        self.verbose = verbose

        self.trainx = []
        self.trainy = []
        self.testx = []
        self.testy = []

    def get_mse_from_predictions(self, adaptive=True, window_size=None):
        """
            Iterates the test data getting predictions for each timestep. Returns
            the MSE over all predictions.

            adaptive - if true, trains new networks as data become available
            window_size - size of window for current variance and mean values
        """
        if window_size is None:
            window_size = self.segment_size
        histx, histy = []
        predictions = []
        win_size = window_size
        for i in range(self.testsize - 1):
            idx = self.trainsize + self.look_back - win_size + i + 1
            window = self.x[idx:idx + win_size]
            xp, yp = self.testx[i], self.testy[i]
            histx.append(xp)
            histy.append(yp)
            if adaptive and len(histx) == self.segment_size:
                # train new network
                histx, histy = [], []
            prediction = self.get_prediction(xp, window.mean(), window.var())
            predictions.append(prediction)

        return mse(self.testy.tolist(), predictions)


    def get_prediction(self, x, mean, variance):
        """
            Gets the prediction from the Ensemble

            x - input
            mean - sliding window mean
            variance - sliding window variance
        """
        prediction = 0
        weights = []
        for method in self.methods:
            distance = method.get_distance(mean, variance)
            weight = 1/(distance**2)
            weights.append(weight)
            prediction += (method.get_prediction(x).reshape(1)*weight)
            print(method.get_prediction(x).reshape(1), method.variance, method.mean, variance, mean, distance, weight)
        print('Prediction:', prediction/sum(weights))
        return prediction / sum(weights)


    def create_methods(self, batch_size=100, verbose=2, epochs=1850):
        """
            Creates the methods that make up the ensemble. The method class
            includes the methods variance and mean so the data must be defined
            first.
        """
        print()
        if not self.data_set:
            print("Must set data on ensemble before creating the methods.\n\
                   \tensemble.set_data_from_file(filename)\n\
                   \tensemble.create_datasets()")
            exit()

        for i in range(len(self.trainx)):
            x, y = self.trainx[i], self.trainy[i]
            lstm = MyLSTM(x.shape[1], 4, [27, 25, 3, 45], 1, epochs=epochs,
                          batch_size=batch_size, fit_verbose=verbose,
                          variables=x.shape[2])
            self.methods.append(Method(lstm, x.mean(), x.var()))
        return


    def create_datasets(self):
        """
            Segments the data based on the ensemble's training style
        """
        train = self.x[:self.trainsize+self.look_back]
        test = self.x[self.trainsize+1:]

        trainx, trainy = self.create_lookback_dataset(train)
        testx, testy = self.create_lookback_dataset(test)

        if self.train_style == 'sequential':
            trainx = self.chunk_data(trainx)
            trainy = self.chunk_data(trainy)
        elif self.train_style == 'random':
            print("'random' training not yet implemented.")
            exit()
        elif self.train_style == 'overlap':
            trainx = [trainx[(i*self.shift):(self.base_size + (i*self.shift))]
                      for i in range(self.num_segments)]
            trainy = [trainy[(i*self.shift):(self.base_size + (i*self.shift))]
                      for i in range(self.num_segments)]
        else:
            print("Invalid training style for ensemble.")
            exit()

        self.trainx, self.trainy = trainx, trainy
        self.testx, self.testy = testx, testy
        return


    def train_methods(self):
        """
            Trains the methods in the ensemble.
        """
        if len(self.methods) == 0:
            print("No methods to train! Create the methods first:\n\
                   \tensemble.create_methods()")
            exit()

        for i in range(len(self.methods)):
            print('Training method', i+1, 'out of', len(self.methods), '...')
            x, y = self.trainx[i], self.trainy[i]
            method = self.methods[i]
            method.train(x, y)
        return


    def train_or_load(self):
        """
            TODO: call from 'train_methods' -- check for files of weights, train
                  if they don't exist
        """
        return


    def set_data_from_file(self, filename):
        """
            Sets the x and y data from the given filename.
        """
        self.dh = DataHandler(filename)
        self.dh.timeSeriesToSupervised()
        self.data = self.dh.tsdata.values
        self.x, self.y = self.data[:,1], self.data[:,3]
        self.testsize = len(self.data) - self.trainsize - self.look_back
        self.data_set = True


    def chunk_data(self, data):
        """
            Splits data into n chunks, all evenly sized except (perhaps) the last one.
        """
        n = self.segment_size
        return [data[i:i+n] for i in range(0, data.shape[0], n)]


    def create_lookback_dataset(self, data):
        """
            Builds arrays containing previous information for the x inputs.
        """
        lookback = self.look_back

        datax, datay = [], []
        if lookback == 0:
            lookback = 1
        rng = len(data)-lookback
        for i in range(rng):
            datax.append(data[i:i+lookback])
            datay.append(data[i + lookback])
        arr = np.asarray(datax)
        return arr.reshape(arr.shape[0], 1, arr.shape[1]), np.asarray(datay)
