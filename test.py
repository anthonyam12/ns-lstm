from data_handler import *
import numpy as np

def create_lookback_dataset(trainx, lookback):
    datax, datay = [], []
    for i in range(len(trainx) - lookback):
        datax.append(trainx[i:i+lookback])
        datay.append(trainx[i + lookback])
    arr = np.asarray(datax)
    return arr.reshape(arr.shape[0], 1, arr.shape[1]), np.asarray(datay)


if __name__ == '__main__':
    dh = DataHandler('./data/Sunspots.csv')
    dh.timeSeriesToSupervised()

    data = dh.tsdata.values[:500]
    x, y = data[:, 1], data[:, 3]
    trainsize = 200
    look_back = 100

    testsize = len(data) - trainsize
    trainx, trainy = x[:trainsize + look_back], y[look_back:trainsize + look_back]
    testx, testy = x[trainsize + look_back:], y[trainsize + look_back:]
    testx, testy = testx.tolist(), testy.tolist()

    trainx, trainy = create_lookback_dataset(trainx, look_back)
    print(trainx, trainy)
    print(trainx.shape, trainy.shape)
