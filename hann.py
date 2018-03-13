from data_handler import *
from errors import *
from ann import *
from lstm import *
import numpy as np

def chunk_data(data, n):
    """
        Splits data into n chunks, all evenly sized except (perhaps) the last one.
    """
    return [data[i:i+n] for i in range(0, len(data), n)]


def calc_variance(data):
    var = np.var(data, dtype=np.float64)
    return var


def train_networks(trainx, trainy):
    """
        Create and train the ensemble of neural networks.
    """
    networks = []
    for i in range(len(trainx)):
        x, y = trainx[i], trainy[i]

        x = np.asarray(x)
        x = x.reshape(x.shape[0], 1, 1)
        print("Training network", i + 1, "out of", len(trainx), "...")
        networks.append(MyLSTM(1, 15, [10 for _ in range(15)], 1, epochs=450,
                            batch_size=100, fit_verbose=0))
        networks[i].train(x, y)


if __name__ == '__main__':
    dh = DataHandler('./data/Sunspots.csv')
    dh.timeSeriesToSupervised()

    data = dh.tsdata.values
    x, y = data[:, 1], data[:, 3]
    trainsize = 2000# int(len(data)*.66)
    testsize = len(data) - trainsize
    trainx, trainy = x[:trainsize], y[:trainsize]
    testx, testy = x[trainsize:], y[trainsize:]
    testx, testy = testx.tolist(), testy.tolist()

    chunksize = int(trainsize/20)
    trainx = chunk_data(trainx.tolist(), chunksize)
    trainy = chunk_data(trainy.tolist(), chunksize)

    train_networks(trainx, trainy)

    histx, histy = [], []
    for i in range(0, testsize):
        xp, yp = testx[i], testy[i]
        histx.append(xp)
        histy.append(yp)
        if len(histx) == chunksize:
            # train new nn
            histx, histy = [], []

    """
        TODO:
            - break data up into chunks (size? <- part of the problem)
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
