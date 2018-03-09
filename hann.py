from data_handler import *
from errors import *
from ann import *

if __name__ == '__main__':
    dh = DataHandler('./data/Sunspots.csv')
    dh.timeSeriesToSupervised()

    data = dh.tsdata.values
    x, y = data[:, 1], data[:, 3]

    # ANN(self, input_size, num_hidden_layers, hidden_layer_sizes, output_size,
    #                  epochs=50, batch_size=1, fit_verbose=2):
    # model = ANN()

    """
        TODO:
            - beak data up into chunks (size? <- part of the problem)
                - differing chunk sizes?
            - create ANN for each chunk of data
            - find statistical properties of each ANN's testing data (function we're trying to learn)
                - create dictionary of statistical values -> ANN
            - implement k-NN (or another ANN) to accumulate predictions of the ANNs
              that are near the current statistical properties of the time series
                - to find these properties just use the last 'n' values of the
                  series (even if used by ANN to train)
            - train new NN everytime there are enough points in the new 'history' set
                - if we use 100 values per NN, predict 100 points out and train a new NN
    """
