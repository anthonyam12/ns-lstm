from errors import *
from data_handler import *
from lstm import *
import time
import numpy as np


if __name__ == '__main__':
    # random.seed(time.time)

    dh = DataHandler('./data/Sunspots.csv')
    dh.timeSeriesToSupervised()
    dh.splitData(len(dh.tsdata)-2000, 2000, 0)

    train, test, out = dh.getDataSets()

    trainx, trainy = train[:, 1], train[:, 3]
    testx, testy = test[:, 1], test[:, 3]
    trainy = trainy.reshape(trainy.shape[0], 1)
    trainx = trainx.reshape(trainx.shape[0], 1)
    testx = testx.reshape(testx.shape[0], 1)
    trainx = trainx.reshape((trainx.shape[0], 1, trainx.shape[1]))
    testx = testx.reshape((testx.shape[0], 1, testx.shape[1]))

    # lstm = MyLSTM(trainx.shape[1], 11, [32, 35, 41, 13, 32, 50, 34, 7, 38, 23, 50],
    #               trainy.shape[1], epochs=981, fit_verbose=2,
    #               batch_size=100)
    lstm = MyLSTM(trainx.shape[1], 11, [32, 35, 41, 13, 32, 50, 34, 7, 38, 23, 50], 1,
                  epochs=750, batch_size=100,
                  fit_verbose=2, variables=trainx.shape[2])
    lstm.train(trainx, trainy)
    y_hat = lstm.predict(testx)
    errors = testy - y_hat[:, 0]
    print(mse(testy, y_hat))
    print(mae(testy, y_hat))
    # lstm.save_model_weights('./weights/benchmarks/sunspots.h5')
