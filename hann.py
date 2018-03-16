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


def create_datasets(trainx, trainy, look_back=100):
    """
        Builds arrays containing previous information for the x inputs.
    """
    datax, datay = [], []
    print(len(trainx))
    for i in range(len(trainx)-look_back - 1):
        datax.append(trainx[i:(i+look_back)])
        datay.append(trainy[i + look_back])
    return np.asarray(datax), np.asarray(datay)


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
        networks.append([MyLSTM(1, 11, [30 for _ in range(15)], 1, epochs=850,
                            batch_size=100, fit_verbose=0), calc_variance(x)])
        networks[i][0].train(x, y)
    return networks


if __name__ == '__main__':
    dh = DataHandler('./data/Sunspots.csv')
    dh.timeSeriesToSupervised()

    data = dh.tsdata.values
    x, y = data[:, 1], data[:, 3]
    trainsize = 200 # Ensure this is a minimum of lookback*(# chunks) for the lookback
                    # on the LSTM
    testsize = len(data) - trainsize
    trainx, trainy = x[:trainsize], y[:trainsize]
    testx, testy = x[trainsize:], y[trainsize:]
    testx, testy = testx.tolist(), testy.tolist()

    chunksize = int(trainsize/1)
    trainx = chunk_data(trainx.tolist(), chunksize)
    trainy = chunk_data(trainy.tolist(), chunksize)

    trainx[0], trainy[0] = create_datasets(trainx[0], trainy[0])
    trainx[0] = trainx[0].reshape(trainx[0].shape[0], 1, trainx[0].shape[1])

    print(len(trainx[0]), len(trainy[0]))
    # networks = train_networks(trainx, trainy)

# ###!!!!!!!!!!!!!!!!!!!!!!!!!! BUILDING LSTMs WRONG!! need to have longer sequence as input to predict
#     histx, histy = [], []
#     predictions = []
#     var_window = x[(trainsize-100):]
#     print(len(var_window))
#     prd = predict(np.asarray(testx[0], var_window).reshape(1, 1, 100))
    # for i in range(0, testsize):
    #     xp, yp = testx[i], testy[i]
    #     histx.append(xp)
    #     histy.append(yp)
    #     xp = np.asarray(xp).reshape(1, 1, 1)
    #     if len(histx) == chunksize:
    #         # train new nn
    #         histx, histy = [], []
    #     prediction = 0
    #     for nn in networks:
    #         prediction += nn[0].predict(xp)
    #         print(prediction, yp)
    #
    # print(len(var_window))
    # print(mse(predictions, testy))
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
