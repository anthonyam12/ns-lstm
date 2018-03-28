from data_handler import *
from errors import *
from ann import *
from lstm import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
    if lookback == 0:
        lookback = 1
    for i in range(len(trainx) - lookback):
        datax.append(trainx[i:i+lookback])
        datay.append(trainx[i + lookback])
    arr = np.asarray(datax)
    return arr.reshape(arr.shape[0], 1, arr.shape[1]), np.asarray(datay)


def calc_variance(data):
    var = np.var(data, dtype=np.float64)
    return var


def train_networks(trainx, trainy, lookback):
    """
        Create and train the ensemble of neural networks.
    """
    global colors, m
    networks = []
    for i in range(len(trainx)):
        # Update plot
        plt.gca().add_patch(
                patches.Rectangle(((i*chunk_size) + lookback, 0),
                              chunk_size, m + 8,
                              fill=False, ls='dashed', ec=colors[i%len(colors)])
        )
        plt.text((i*chunk_size)+lookback+10, m+15, '$f_{' + str(i+1) + '}^*$')
        plt.pause(0.0001)
        plt.draw()

        x, y = trainx[i], trainy[i]
        print("Training network", i + 1, "out of", len(trainx), "...")
        networks.append([MyLSTM(x.shape[1], 4, [27, 25, 3, 45], 1, epochs=550,
                            batch_size=100, fit_verbose=2, variables=x.shape[2]),
                            calc_variance(x)])
        networks[i][0].train(x, y)
    return networks


if __name__ == '__main__':
    global colors, m
    colors = ['red', 'blue', 'green']
    trainsize = 2000
    look_back = 200
    num_chunks = 2
    chunk_size = int(trainsize / num_chunks)

    dh = DataHandler('./data/Sunspots.csv')
    dh.timeSeriesToSupervised()

    data = dh.tsdata.values
    x, y = data[:, 1], data[:, 3]
    m = max(x)

    plt.plot(x)
    plt.ylim(0, m + 35)
    plt.xlabel('Timestep (day)')
    plt.ylabel('Number Sunspots')
    plt.title('Sunspot data with data subset\ncurrently under consideration')
    plt.ion() # non-blocking
    plt.show()

    testsize = len(data) - trainsize
    train = x[:trainsize + look_back]
    test  = x[(trainsize - look_back + 1):]

    trainx, trainy = create_lookback_dataset(train, look_back)
    trainx = chunk_data(trainx, chunk_size)
    trainy = chunk_data(trainy, chunk_size)
    networks = train_networks(trainx, trainy, look_back)

    testx, testy = create_lookback_dataset(test, look_back)
    histx, histy = [], []
    predictions = []
    # TODO: Different window now since lookback
    var_window = x[(trainsize-100):]
    win_size = 100 # doesn't need to be 100, just for demo
    for i in range(testsize-1):
        # Works but is slow and kind of annoying
        # var_rect = patches.Rectangle(((trainsize-win_size)+i, 0), win_size, m + 8,
        #               fill=False, ls='solid', ec='black')
        # plt.gca().add_patch(var_rect)
        # plt.pause(.0001)
        # plt.draw()

        xp, yp = testx[i], testy[i]
        histx.append(xp)
        histy.append(yp)
        xp = xp.reshape((1, xp.shape[0], xp.shape[1]))
        if len(histx) == chunk_size:
            # train new nn
            # trx, ty = create_lookback_dataset(histx, look_back)
            # trx = chunk_data(trx, chunk_size)
            # ty = chunk_data(ty)
            # print(len(trx), len(trx[0]))
            histx, histy = [], []
        prediction = 0
        for nn in networks:
            p = nn[0].predict(xp)
            prediction += p.reshape(1)
        prediction /= len(networks)
        predictions.append(prediction.tolist())

        # var_rect.remove()
        # plt.pause(.0001)
        # plt.draw()
        # print(prediction, yp, len(networks))

    print(predictions)
    print(mse(testy.tolist(), predictions))
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
