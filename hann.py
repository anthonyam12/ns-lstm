from data_handler import *
from errors import *
from ann import *
from lstm import *
import numpy as np

def chunk_data(data, n):
    """
        Splits data into n chunks, all evenly sized except (perhaps) the last one.
    """
    return [data[i:i+n] for i in range(0, data.shape[0], n)]


def create_lookback_dataset(trainx, lookback):
    """
        Builds arrays containing previous information for the x inputs.
    """
    datax, datay = [], []
    for i in range(len(trainx) - lookback):
        datax.append(trainx[i:i+lookback])
        datay.append(trainx[i + lookback])
    arr = np.asarray(datax)
    return arr.reshape(arr.shape[0], 1, arr.shape[1]), np.asarray(datay)


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

        print("Training network", i + 1, "out of", len(trainx), "...")
        networks.append([MyLSTM(x.shape[1], 5, [10 for _ in range(10)], 1, epochs=550,
                            batch_size=200, fit_verbose=2, variables=x.shape[2]),
                            calc_variance(x)])
        networks[i][0].train(x, y)
    return networks


if __name__ == '__main__':
    trainsize = 2000
    look_back = 300
    num_chunks = 3
    chunk_size = int(trainsize / num_chunks)

    dh = DataHandler('./data/Sunspots.csv')
    dh.timeSeriesToSupervised()

    data = dh.tsdata.values
    x, y = data[:, 1], data[:, 3]

    testsize = len(data) - (trainsize + look_back)
    trainx = x[:trainsize + look_back]
    testx, testy = x[(trainsize + 1):], y[(trainsize + 1):]
    # testx, testy = testx.tolist(), testy.tolist()

    trainx, trainy = create_lookback_dataset(trainx, look_back)
    trainx = chunk_data(trainx, chunk_size)
    trainy = chunk_data(trainy, chunk_size)
    networks = train_networks(trainx, trainy)

    testx, _ = create_lookback_dataset(testx, look_back)
    histx, histy = [], []
    predictions = []
    # TODO: Different window now since lookback
    var_window = x[(trainsize-100):]
    for i in range(100, testsize):
        xp, yp = testx[i], testy[i]
        histx.append(xp)
        histy.append(yp)
        xp = xp.reshape((1, xp.shape[0], xp.shape[1]))
        if len(histx) == chunk_size:
            # train new nn
            histx, histy = [], []
        prediction = 0
        for nn in networks:
            prediction += nn[0].predict(xp)
            print(prediction, yp)

    print(mse(predictions, testy))
    """
        TODO:
            - break data up into chunks (size? <- part of the problem)
                - differing chunk sizes?
                - overlapping chunk sizes?
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
