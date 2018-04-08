from data_handler import *
from errors import *
from lstm import *
from ensemble import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def chunk_data(data, n):
    """
        Splits data into n chunks, all evenly sized except (perhaps) the last one.
    """
    return [data[i:i+n] for i in range(0, data.shape[0], n)]


def create_lookback_dataset(data, lookback, test=False):
    """
        Builds arrays containing previous information for the x inputs.
    """
    datax, datay = [], []
    if lookback == 0:
        lookback = 1
    rng = len(data)-lookback
    for i in range(rng):
        datax.append(data[i:i+lookback])
        datay.append(data[i + lookback])
    arr = np.asarray(datax)
    return arr.reshape(arr.shape[0], 1, arr.shape[1]), np.asarray(datay)


def train_networks(trainx, trainy, lookback):
    """
        Create and train the ensemble of neural networks.
    """
    global colors, m, ensemble
    for i in range(len(trainx)):
        # Update plot
        plt.gca().add_patch(
                patches.Rectangle(((i*chunk_size) + lookback, 0),
                              chunk_size, m + 8, fill=False, ls='dashed', ec=colors[i%len(colors)]))
        plt.text((i*chunk_size)+lookback+10, m+15, '$f_{' + str(i+1) + '}^*$')
        plt.pause(0.0001)
        plt.draw()

        x, y = trainx[i], trainy[i]
        print("Training network", i + 1, "out of", len(trainx), "...")
        lstm = MyLSTM(x.shape[1], 4, [27, 25, 3, 45], 1, epochs=1850,
                      batch_size=100, fit_verbose=2, variables=x.shape[2])
        lstm.train(x, y)
        ensemble.add_method(Method(lstm, x.mean(), x.var()))


if __name__ == '__main__':
    global colors, m, ensemble
    ensemble = Ensemble()
    colors = ['red', 'blue', 'green']
    trainsize = 2000
    look_back = 1  # don't use lookback = 0!!! just use 1 (gives bad indices in test arrays)
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
    plt.ion()

    testsize = len(data) - trainsize - look_back
    train = x[:trainsize + look_back]
    test  = x[trainsize+1:]

    trainx, trainy = create_lookback_dataset(train, look_back)
    trainx = chunk_data(trainx, chunk_size)
    trainy = chunk_data(trainy, chunk_size)
    train_networks(trainx, trainy, look_back)

    testx, testy = create_lookback_dataset(test, look_back, True)
    histx, histy = [], []
    predictions = []
    win_size = chunk_size # doesn't need to be 100, just for demo
    for i in range(testsize-1):
        idx = trainsize+look_back-win_size+i+1
        window = x[idx:idx+win_size]
        # Works but is slow and kind of annoying
        # var_rect = patches.Rectangle((idx, 0), win_size, m + 8,
        #               fill=False, ls='solid', ec='black')
        # plt.gca().add_patch(var_rect)
        # plt.pause(.0000001)
        # plt.draw()

        xp, yp = testx[i], testy[i]
        histx.append(xp)
        histy.append(yp)
        print("TARGET", yp)
        xp = xp.reshape((1, xp.shape[0], xp.shape[1]))
        if len(histx) == chunk_size:
            # train new nn
            histx, histy = [], []
        prediction = ensemble.get_prediction(xp, window.mean(), window.var())
        predictions.append(prediction.tolist())

        # var_rect.remove()
        # plt.draw()
        # print(prediction, yp, len(networks))

    print(predictions)
    print(mse(testy.tolist(), predictions))
