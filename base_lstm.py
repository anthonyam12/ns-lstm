from errors import *
from data_handler import *
from lstm import *
import time
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # random.seed(time.time)

    dh = DataHandler('./data/EURUSD.csv')
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

    # Sunspots
    # lstm = MyLSTM(trainx.shape[1], 11, [32, 35, 41, 13, 32, 50, 34, 7, 38, 23, 50], 1,
    #               epochs=750, batch_size=100,
    #               fit_verbose=2, variables=trainx.shape[2])
    # Mackey
    # lstm = MyLSTM(trainx.shape[1], 9, [28,4,36,47,2,4,5,1,37], 1,
    #               epochs=702, batch_size=100,
    #               fit_verbose=2, variables=trainx.shape[2])
    # EUR USD
    lstm = MyLSTM(trainx.shape[1], 7, [8,38,46,31,49,14,14], 1,
                  epochs=300, batch_size=200,
                  fit_verbose=2, variables=trainx.shape[2])

    lstm.train(trainx, trainy)
    # lstm.load_model_weights('./weights/benchmarks/sunspots.h5')
    y_hat = lstm.predict(testx)
    errors = testy - y_hat[:, 0]
    print(mse(testy, y_hat))
    print(mae(testy, y_hat))
    lstm.save_model_weights('./weights/benchmarks/eurusd.h5')

    plt.plot(testy, label='actual')
    plt.plot(y_hat, label='predicted')
    plt.xlabel('Timestep')
    plt.ylabel('EUR/USD Exchange Rate')
    plt.title('Actual vs. LSTM EUR/USD Forex Test Set')
    plt.legend()
    plt.savefig('./graphs/results/lstm_results_eurusd.eps', dpi=1000)
    plt.show()
